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
CUDA_PATH ?= /usr/local/cuda
#CUDA_PATH ?= /cvmfs/soft.computecanada.ca/easybuild/software/2020/Core/cudacore/11.7.0
# CUDA_PATH ?= /cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Core/cudacore/12.2.2
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
CUSPARSELT_PATH ?= /home/kobeliu/libraries/libcusparse_lt


# internal flags
NVCCFLAGS   := -m$(shell getconf LONG_BIT)
CCFLAGS     :=
LDFLAGS     :=

ALL_CCFLAGS :=
ALL_CCFLAGS += $(NVCCFLAGS)
ALL_CCFLAGS += $(addprefix -Xcompiler ,$(CCFLAGS))

ALL_LDFLAGS :=
ALL_LDFLAGS += $(ALL_CCFLAGS)
ALL_LDFLAGS += $(addprefix -Xlinker ,$(LDFLAGS))

# Common includes and paths for CUDA
INCLUDES  := -I$(CUDA_PATH)/include/
LIBRARIES := -lcublas 
LIBRARIES += -lcusparse 
LIBRARIES += -L${CUDA_PATH}/lib64

#
ifneq ($(SPUTNIK_PATH),)
	INCLUDES  += -I${SPUTNIK_PATH}
	LIBRARIES += -L${SPUTNIK_PATH}/build/sputnik/ -lsputnik -lglog 
endif
#
ifneq ($(SPMM_PATH),)
INCLUDES  += -I${SPMM_PATH}/build
LIBRARIES += -L${SPMM_PATH}/build -lSpMM_API
endif

ifneq ($(USE_CUSPARSELT),)
INCLUDES  += -I${CUSPARSELT_PATH}/include
LIBRARIES += -lcusparseLt -L${CUSPARSELT_PATH}/lib
endif
################################################################################

# Gencode arguments
SMS ?= 80
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

ALL_CCFLAGS += --threads 0 --std=c++11 -O3 -lineinfo
ALL_CCFLAGS += -maxrregcount=255
ALL_CCFLAGS += --use_fast_math
ALL_CCFLAGS += --ptxas-options=-v,-warn-lmem-usage,--warn-on-spills
################################################################################
# Target rules
all: spmm_test


spmm_test.o:  spmm_test.cu spmm_test_utils.h sparTA.h spmm_decompose.h FSP_Computation.h FSP_Batch8.h
	$(EXEC) $(NVCC) $(INCLUDES) $(ALL_CCFLAGS) $(GENCODE_FLAGS) -o $@ -c $<

spmm_test: spmm_test.o
	$(EXEC) $(NVCC) $(ALL_LDFLAGS) $(GENCODE_FLAGS) -o $@ $+ $(LIBRARIES) 

clean:
	rm -f spmm_test spmm_test.o