#-------------------------------------------------
#
# Project created by QtCreator 2019-08-05T18:35:53
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = parseGreen
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

# Define output directories
DESTDIR = ./
CUDA_OBJECTS_DIR = OBJECTS_DIR/../cuda

SOURCES += \
        main.cpp \
        mainwindow.cpp

HEADERS += \
        mainwindow.h

FORMS += \
      mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resource.qrc

CUDA_SOURCES += extractgreencuda.cu

# MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
MSVCRT_LINK_FLAG_DEBUG   = "/MDd"
MSVCRT_LINK_FLAG_RELEASE = "/MD"

CUDA_DIR = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1"
CUDA_SDK = "C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1"

CUDA_ARCH = sm_61 # as supported by the GeForce GTX 1050
SYSTEM_NAME = x64
SYSTEM_TYPE = 64
NVCC_OPTIONS = --use_fast_math

INCLUDEPATH += \
            $$CUDA_DIR/include \
            $$CUDA_SDK/common/inc \
            $$CUDA_SDK/../shared/inc
QMAKE_LIBDIR += \
             $$CUDA_DIR/lib/$$SYSTEM_NAME \
             $$CUDA_SDK/common/lib/$$SYSTEM_NAME \
             $$CUDA_SDK/../shared/lib/$$SYSTEM_NAME

QMAKE_LFLAGS_RELEASE = /NODEFAULTLIB:msvcrtd.lib

#CUDA_LIBS = cudart cuda
#CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')
#NVCC_LIBS = $$join(CUDA_LIBS,' -l','-l', '')
#LIBS += $$join(CUDA_LIBS,'.lib ', '', '.lib')
#LIBS += -L$$CUDA_DIR/lib/$$SYSTEM_NAME -lcudart -lcuda -lcudadevrt


# The following makes sure all path names (which often include spaces) are put between quotation marks
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# Add the necessary libraries
CUDA_LIB_NAMES = cudart_static kernel32 user32 gdi32 winspool comdlg32 \
                 advapi32 shell32 ole32 oleaut32 uuid odbc32 odbccp32 \
                 #freeglut glew32

for(lib, CUDA_LIB_NAMES) {
    CUDA_LIBS += -l$$lib
}
LIBS += $$CUDA_LIBS



# Configuration of the Cuda compiler
CONFIG(debug, debug|release) {
    # Debug mode
    cuda_d.input = CUDA_SOURCES
    cuda_d.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda_d.commands = $$CUDA_DIR/bin/nvcc.exe -D_DEBUG $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                      --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                      --compile -cudart static -g -DWIN32 -D_MBCS \
                      -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/Od,/Zi,/RTC1" \
                      -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG \
                      -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda_d.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
    # Release mode
    cuda.input = CUDA_SOURCES
    cuda.output = $$CUDA_OBJECTS_DIR/${QMAKE_FILE_BASE}_cuda.obj
    cuda.commands = $$CUDA_DIR/bin/nvcc.exe $$NVCC_OPTIONS $$CUDA_INC $$LIBS \
                    --machine $$SYSTEM_TYPE -arch=$$CUDA_ARCH \
                    --compile -cudart static -DWIN32 -D_MBCS \
                    -Xcompiler "/wd4819,/EHsc,/W3,/nologo,/O2,/Zi" \
                    -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE \
                    -c -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
    cuda.dependency_type = TYPE_C
    QMAKE_EXTRA_COMPILERS += cuda
}

