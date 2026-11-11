# Tiny 分支操作记录

2026-11-11接手时核查远端：portable.zip和9份完整依赖存在，torch CPU wheel为146800640/194889522 bytes；没有tiny setup/训练进程或root status。因此沿用分支的有界chunk上传脚本，从远端现有字节续传，而未重复启动已有任务。续传12块，最后不足4MiB，总158.58秒。

上传完成后启动私有setup，启动PID2678861；入口脚本拒绝已有status和输出。setup从同一portable.zip安全解包，清除Python包路径环境变量和LD_LIBRARY_PATH，新建不继承系统包的venv，使用离线wheelhouse安装10个锁定运行依赖。运行全程CPU、2线程，CUDA_VISIBLE_DEVICES为空，未修改共享Python。

同主机记录setup总约43.9秒，pipeline约9.5秒，exit_code0。三模型tiny训练与所有检查一次成功；pip check、summary重建及共享torch版本断言通过。setup与结果包内日志全部本地保存，脚本/认证临时Jupyter kernel的cleanup为HTTP204。认证仅从环境传入，无token落盘。

下载25文件结果ZIP并按workspace边界验证后解包。独立审计首次因Windows Path转换POSIX远端路径导致assert失败，此为本地审计路径检查问题，未重新运行训练；改用原始POSIX字符串前缀后全部审计PASS，过程已在RESULTS中披露。

保留：原上传中间回执、完整上传回执、启动/监控/结果打包回执、portable.zip和全部依赖、run1全部原始结果与三个小checkpoint。没有删除环境或原始结果，未启动驻留服务。
