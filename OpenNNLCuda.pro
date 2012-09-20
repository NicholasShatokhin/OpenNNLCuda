TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp
CUDA_SOURCES += opennnl.cu

HEADERS += \
    opennnl.h \
    utils.h


CUDA_DIR = /usr/local/cuda

CUDA_ARCH = sm_11

NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

QMAKE_LIBDIR += $$CUDA_DIR/lib

LIBS += -lcudart

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

cuda.commands = $$CUDA_DIR/bin/nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

QMAKE_EXTRA_COMPILERS += cuda
