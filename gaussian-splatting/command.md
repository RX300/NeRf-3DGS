训练常用命令：
python train.py -s D:\A_study\nerf3dgs\dataset\nerf_synthetic-20230812T151944Z-001\nerf_synthetic\hotdog --iterations 5000 --ip 127.0.0.1 --port 6009

python train.py -s D:\A_study\nerf3dgs\dataset\tiny_nerf_data --iterations 5000 --ip 127.0.0.1 --port 6009
./`<SIBR install dir>`/bin/SIBR_remoteGaussian_app

渲染常用命令：
D:\A_study\nerf3dgs\viewers\bin\SIBR_gaussianViewer_app.exe -m D:\A_study\nerf3dgs\gaussian-splatting\output\ca119e46-9

D:\A_study\nerf3dgs\viewers\bin\SIBR_remoteGaussian_app.exe --ip 127.0.0.1 --port 6009

可能会用到的第三方教程：
https://blog.csdn.net/x5445687d/article/details/143231814 - 使用windows预编译的 SIBR Viewer 实时查看远程服务器训练结果
