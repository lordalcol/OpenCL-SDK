/*
 * Copyright (c) 2026 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenCL SDK includes
#include <CL/Utils/Context.hpp>
#include <CL/Utils/Device.hpp>
#include <CL/Utils/File.hpp>
#include <CL/SDK/Context.hpp>
#include <CL/SDK/Options.hpp>
#include <CL/SDK/CLI.hpp>
#include <CL/SDK/Random.hpp>

// STL includes
#include <iostream>
#include <random>
#include <tuple> // std::make_tuple
#include <vector>

// TCLAP includes
#include <tclap/CmdLine.h>

// Sample-specific options
struct CommandBufferOptions
{
    size_t length;
    size_t iterations;
};

// Add options to CLI parsing SDK utility
template <> auto cl::sdk::parse<CommandBufferOptions>()
{
    return std::make_tuple(std::make_shared<TCLAP::ValueArg<size_t>>(
                               "l", "length", "Length of input", false,
                               1'048'576, "positive integral"),
                           std::make_shared<TCLAP::ValueArg<size_t>>(
                               "i", "iterations",
                               "Number of times to run the command buffer",
                               false, 10, "positive integral"));
}
template <>
CommandBufferOptions cl::sdk::comprehend<CommandBufferOptions>(
    std::shared_ptr<TCLAP::ValueArg<size_t>> length_arg,
    std::shared_ptr<TCLAP::ValueArg<size_t>> iterations_arg)
{
    return CommandBufferOptions{ length_arg->getValue(),
                                 iterations_arg->getValue() };
}

int main(int argc, char* argv[])
{
    try
    {
        // Parse command-line options
        auto opts = cl::sdk::parse_cli<cl::sdk::options::Diagnostic,
                                       cl::sdk::options::SingleDevice,
                                       CommandBufferOptions>(argc, argv);
        const auto& diag_opts = std::get<0>(opts);
        const auto& dev_opts = std::get<1>(opts);
        const auto& cmdbuf_opts = std::get<2>(opts);

        // Create runtime objects based on user preference or default
        cl::Context context = cl::sdk::get_context(dev_opts.triplet);
        cl::Device device = context.getInfo<CL_CONTEXT_DEVICES>().at(0);

        // If the selected device can't run command buffers, use the first
        // device that can
        if (!cl::util::supports_extension(device, "cl_khr_command_buffer"))
        {
            std::cout << device.getInfo<CL_DEVICE_NAME>()
                      << " does not support cl_khr_command_buffer."
                      << std::endl;
            std::vector<cl::Platform> platforms;
            cl::Platform::get(&platforms);
            device = cl::Device{};
            for (const auto& p : platforms)
            {
                std::vector<cl::Device> devices;
                p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
                for (const auto& d : devices)
                    if (!device()
                        && cl::util::supports_extension(
                            d, "cl_khr_command_buffer"))
                        device = d;
            }
            if (!device())
            {
                std::cout << "No device supports it." << std::endl;
                return EXIT_SUCCESS;
            }
            context = cl::Context{ device };
        }
        cl::Platform platform{ device.getInfo<CL_DEVICE_PLATFORM>() };

        if (!diag_opts.quiet)
        {
            std::cout << "Selected platform: "
                      << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n"
                      << "Selected device: " << device.getInfo<CL_DEVICE_NAME>()
                      << "\n"
                      << std::endl;
        }

        // Some devices need queue properties to run command buffers
        cl::CommandQueue queue{
            context, device,
            device.getInfo<
                CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR>()
        };

        // Compile kernels
        cl::Program program{
            context, cl::util::read_exe_relative_text_file("commandbuffer.cl")
        };
        program.build(device);
        cl::Kernel square{ program, "square" }, add{ program, "add" };

        // Initialize host-side storage
        const auto length = cmdbuf_opts.length;
        const auto bytes = length * sizeof(cl_int);

        auto prng = [engine = std::default_random_engine{},
                     dist = std::uniform_int_distribution<cl_int>{
                         -1000, 1000 }]() mutable { return dist(engine); };

        std::vector<cl_int> arr_x(length), arr_y(length), arr_out(length);
        cl::sdk::fill_with_random(prng, arr_y);

        // Initialize device-side storage
        cl::Buffer buf_x{ context, CL_MEM_READ_ONLY, bytes },
            buf_y{ context, std::begin(arr_y), std::end(arr_y), true },
            buf_tmp{ context, CL_MEM_READ_WRITE, bytes },
            buf_out{ context, CL_MEM_WRITE_ONLY, bytes };

        // Kernel arguments are captured when a command is recorded
        square.setArg(0, buf_x);
        square.setArg(1, buf_tmp);
        add.setArg(0, buf_tmp);
        add.setArg(1, buf_y);
        add.setArg(2, buf_out);

        // Record both kernels once. Commands in a command buffer have no
        // implicit order, so add waits for square through a sync point.
        cl::CommandBufferKhr command_buffer{ { queue } };
        cl_sync_point_khr square_done;
        command_buffer.commandNDRangeKernel(
            {}, square, cl::NullRange, cl::NDRange{ length }, cl::NullRange,
            nullptr, &square_done);
        std::vector<cl_sync_point_khr> wait_for_square{ square_done };
        command_buffer.commandNDRangeKernel({}, add, cl::NullRange,
                                            cl::NDRange{ length },
                                            cl::NullRange, &wait_for_square);
        command_buffer.finalizeCommandBuffer();

        // Run the command buffer several times, with new input each time
        std::vector<cl::CommandQueue> queues{ queue };
        for (size_t it = 0; it < cmdbuf_opts.iterations; ++it)
        {
            cl::sdk::fill_with_random(prng, arr_x);
            queue.enqueueWriteBuffer(buf_x, CL_FALSE, 0, bytes, arr_x.data());
            command_buffer.enqueueCommandBuffer(queues);
            queue.enqueueReadBuffer(buf_out, CL_TRUE, 0, bytes, arr_out.data());

            for (size_t i = 0; i < length; ++i)
                if (arr_out[i] != arr_x[i] * arr_x[i] + arr_y[i])
                    throw std::runtime_error{ "Verification FAILED!" };
        }

        std::cout << "Ran the command buffer " << cmdbuf_opts.iterations
                  << " times. Verification passed." << std::endl;

        return 0;
    } catch (cl::util::Error& e)
    {
        std::cerr << "OpenCL Utils error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (cl::BuildError& e)
    {
        std::cerr << "OpenCL runtime error: " << e.what() << std::endl;
        for (auto& build_log : e.getBuildLog())
        {
            std::cerr << "\tBuild log for device: "
                      << build_log.first.getInfo<CL_DEVICE_NAME>() << "\n"
                      << std::endl;
            std::cerr << build_log.second << "\n" << std::endl;
        }
        std::exit(e.err());
    } catch (cl::Error& e)
    {
        std::cerr << "OpenCL runtime error: " << e.what() << std::endl;
        std::exit(e.err());
    } catch (std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
