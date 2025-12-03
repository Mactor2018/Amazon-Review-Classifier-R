@echo off
REM ============================================================================
REM BoW 清理脚本 - 删除缓存和生成的文件
REM ============================================================================

echo ============================================
echo BoW Clean Script
echo ============================================
echo.

REM 删除日志文件
echo Cleaning logs...
if exist logs\*.log del /q logs\*.log

REM 删除图片文件
echo Cleaning figures...
if exist figures\*.png del /q figures\*.png

REM 删除模型文件
echo Cleaning model files...
if exist *.model del /q *.model
if exist *.keras del /q *.keras
if exist *.h5 del /q *.h5
if exist *.hd5 del /q *.hd5

REM 删除 RData 缓存
echo Cleaning RData cache...
if exist *.RData del /q *.RData
if exist *.rds del /q *.rds
if exist *.rda del /q *.rda

REM 删除 Rplots
if exist Rplots.pdf del /q Rplots.pdf

echo.
echo ============================================
echo Clean completed!
echo ============================================
pause


