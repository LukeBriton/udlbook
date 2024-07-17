@echo off
setlocal enabledelayedexpansion

REM 设置要处理的目录
set "directory=.\"

REM 切换到目标目录
cd /d "%directory%"

REM 遍历目录及子目录中的所有PDF文件
for /r %%f in (*.pdf) do (
    REM 提取文件名（不含扩展名）
    set "filename=%%~nf"
    REM 获取文件所在的目录路径
    set "filepath=%%~dpf"
    REM 确保输出路径存在
    if not exist "!filepath!" (
        mkdir "!filepath!"
    )
    REM 执行dvisvgm命令
    dvisvgm --pdf --output="!filepath!!filename!.svg" "%%f"
)

endlocal
pause
