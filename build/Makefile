# Copyright 2023 The HeteroSparse Authors. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# host compiler
HOST_COMPILER ?= g++
CUDA_PATH ?= /usr/local/cuda/
#below is the path for Narval
#CUDA_PATH ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.7.0/
# CUDA_PATH ?= /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2/
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     := -fPIC
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I/usr/local/cuda/include/
LIBRARIES := -lcublas -lcusparse

################################################################################

# Gencode arguments
SMS ?= 80
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS +=  --threads 0 --std=c++11 -lineinfo -O3
ALL_CCFLAGS += -maxrregcount=255 
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################

HEAD_FILES = SpMM_API.cuh \
			 ../csrc/Reduction_Kernel.cuh ../csrc/SpMM_Kernel.cuh \
			 ../csrc/MatMulUtilities.cuh \
			 ../csrc/MMA_PTX.cuh ../csrc/AsyncCopy_PTX.cuh \
			 ../csrc/TilingConfig.h
# Target rules
all: libSpMM_API.so

libSpMM_API.so: SpMM_API.o
	$(EXEC) $(NVCC) --shared $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES)

SpMM_API.o:  ../csrc/SpMM_API.cu $(HEAD_FILES)  
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

clean:
	rm -f libSpMM_API.so SpMM_API.o