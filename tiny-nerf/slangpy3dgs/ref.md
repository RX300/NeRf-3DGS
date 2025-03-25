slang Docs:[Automatic Differentiation | slang](https://shader-slang.org/slang/user-guide/autodiff.html#differentiable-type-system)

slangpy Docs:[SlangPy](https://slangpy.shader-slang.org/en/latest/index.html)

slang gaussian:[slang-gaussian-rasterization/slang_gaussian_rasterization/internal/slang at main · google/slang-gaussian-rasterization](https://github.com/google/slang-gaussian-rasterization/tree/main/slang_gaussian_rasterization/internal/slang)

3dGs的代码注释：

[3d gaussian splatting核心代码注释（CUDA部分）_cub::devicescan::inclusivesum-CSDN博客](https://blog.csdn.net/xxxrc5/article/details/135695564)

[3DGS-结合源码做更细节一点的简单解析-3D Gaussian Splatting - 知乎](https://zhuanlan.zhihu.com/p/974410291)

坐标系转换：

[矩阵：行主序、列主序、行向量、列向量 - 知乎](https://zhuanlan.zhihu.com/p/138920694)

[Unity 和 Unreal 渲染中的坐标变换和跨平台兼容 - 知乎](https://zhuanlan.zhihu.com/p/590584851)

在hlsl中，列主序存储的矩阵右乘(矩阵在mul指令的右边)效率高，行主序存储的矩阵左乘效率高。在3dgs的slang中，采取和glsl统一的方式，行主序左乘向量(同时slang的默认存储就是行主序，falcor中slang文件也是按这个顺序来计算的)
