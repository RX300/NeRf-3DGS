  # Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import slangtorch
import os

shaders_path = os.path.dirname(__file__)

TILE_SIZES_HW = [(4,4), (8,8), (16,16)]
print("Loading shaders...")
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6;8.9;8.0;7.5;7.0;6.1;5.2;5.3;5.4;5.5;5.6;5.7;5.8;5.9;5.10'
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
vertex_shader = slangtorch.loadModule(os.path.join(shaders_path, "vertex_shader.slang"),skipNinjaCheck=True)
tile_shader = slangtorch.loadModule(os.path.join(shaders_path, "tile_shader.slang"),skipNinjaCheck=True)
alpha_blend_shaders = {}
for tile_height, tile_width in TILE_SIZES_HW:
  alpha_blend_shaders[(tile_height, tile_width)] = slangtorch.loadModule(os.path.join(shaders_path, "alphablend_shader.slang"), 
                                                                         defines={"PYTHON_TILE_HEIGHT": tile_height, "PYTHON_TILE_WIDTH": tile_width},skipNinjaCheck=True)
print("Loading shaders... done")