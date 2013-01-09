TEMPLATE = app
CONFIG += console
CONFIG -= qt

SOURCES += main.cpp \
    mnistfile.cpp

CUDA_SOURCES += opennnl.cu

OTHER_FILES += opennnl.cu \
    data/mnist/train-labels.idx1-ubyte \
    data/mnist/train-images.idx3-ubyte \
    data/mnist/t10k-labels.idx1-ubyte \
    data/mnist/t10k-images.idx3-ubyte

HEADERS += \
    opennnl.h \
    utils.h \
    cuda_helper.h \
    mnistfile.h \
    LittleBigEndian.h


CUDA_ARCH = sm_40

LIBS += -lcudart

CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

unix {
    CUDA_DIR = /usr/local/cuda
    INCLUDEPATH += /usr/local/cuda/include
    QMAKE_LIBDIR += $$CUDA_DIR/lib

    LIBS += -lcudart

cuda.commands = nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
}

win32 {
  CUDA_DIR = $(CUDA_PATH)
  INCLUDEPATH += $(CUDA_PATH)/include
  QMAKE_LIBDIR += $(CUDA_PATH)/lib/x64

cuda.commands = nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC
}







cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o

#cuda.commands = $$CUDA_DIR/bin/nvcc -g -G -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -G -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}

QMAKE_EXTRA_COMPILERS += cuda
