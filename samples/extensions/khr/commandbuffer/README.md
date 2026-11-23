# Command Buffer Sample

## Sample Purpose

This sample shows how to record OpenCL commands once with the `cl_khr_command_buffer` extension and run them many times. It records two kernels into a command buffer: `square` computes the square of each element of an input buffer, and `add` adds a second buffer to the result. The command buffer is then run several times, with new input written before each run, and the result is checked on the host every time.

Recording the commands once saves the work of setting up and enqueuing each command on every run, which helps applications that run the same sequence of commands many times.

## Key APIs and Concepts

- The device must support `cl_khr_command_buffer`. If it doesn't, the sample says so and exits. The extension is still provisional in the OpenCL headers, so the sample defines `CL_ENABLE_BETA_EXTENSIONS`.
- The command queue is created with the properties the device requires for command buffers (`CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR`).
- Kernel arguments are captured when a kernel command is recorded, so they are set before recording.
- Commands in a command buffer have no implicit order, even when the command buffer is created with an in-order queue. Their dependencies are set with sync points: `add` waits for the sync point that `square` returns.
- After `finalizeCommandBuffer`, no more commands can be recorded, and the command buffer can be enqueued.
- Buffers keep their identity across runs, so new data written to the input buffer is used by the next run.

## Application flow

1. Select a device and check that it supports `cl_khr_command_buffer`.
2. Create a command queue with the required properties, and build the two kernels.
3. Create the buffers and set the kernel arguments.
4. Record `square` and `add` into a command buffer, with a sync point between them, and finalize it.
5. For each iteration: write new input, enqueue the command buffer, read the output and check it.

## Command-line options

- `-p`, `-d`, `-t`: index of the platform, index of the device, and device type to use
- `-l <length>`: length of the buffers (default: 1048576)
- `-i <iterations>`: number of times to run the command buffer (default: 10)

## Used API surface

```c++
cl::util::supports_extension(cl::Device, cl::string)
cl::util::read_exe_relative_text_file(const char*)
cl::sdk::get_context(cl::sdk::options::DeviceTriplet)
cl::CommandQueue(cl::Context, cl::Device, cl_command_queue_properties)
cl::Program(cl::Context, cl::string)
cl::Kernel(cl::Program, const char*)
cl::Buffer(cl::Context, cl_mem_flags, size_type)
cl::CommandBufferKhr(const vector<cl::CommandQueue>&)
cl::CommandBufferKhr::commandNDRangeKernel(...)
cl::CommandBufferKhr::finalizeCommandBuffer()
cl::CommandBufferKhr::enqueueCommandBuffer(vector<cl::CommandQueue>&)
cl::CommandQueue::enqueueWriteBuffer(...)
cl::CommandQueue::enqueueReadBuffer(...)
```
