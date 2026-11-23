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

kernel void square(global const int* x, global int* tmp)
{
    size_t i = get_global_id(0);
    tmp[i] = x[i] * x[i];
}

kernel void add(global const int* tmp, global const int* y, global int* out)
{
    size_t i = get_global_id(0);
    out[i] = tmp[i] + y[i];
}
