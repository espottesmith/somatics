.PHONY: execute clean

ON_CORI           = TRUE
USE_MPI           = FALSE
USE_OMP           = TRUE
USE_MIN_FINDER    = TRUE
USE_TS_FINDER     = TRUE
USE_QHULL         = TRUE
USE_MOLECULE      = TRUE

ifeq ($(USE_MPI), TRUE)
CXX = mpic++
C = mpicc
DEFINES	+= -DUSE_MPI=$(USE_MPI)
else
CXX = g++
C = gcc
endif

ifeq ($(USE_MPI), TRUE)
ifeq ($(ON_CORI), TRUE)
MPIDIR=/global/common/software/m3169/cori/openmpi/4.0.2/gnu/include
CFLAGS += -I$(MPIDIR)
endif
endif

ifeq ($(USE_QHULL), TRUE)
ifeq ($(ON_CORI), TRUE)
QHULLDIR=/global/homes/c/cmcc/.local
CFLAGS += -I$(QHULLDIR)/include
endif
endif

ifeq ($(USE_MOLECULE), TRUE)
ifeq ($(ON_CORI), TRUE)
XTBDIR=/global/homes/e/ewcss/software/grimme/xtb_6.2.2/
CFLAGS += -I$(XTBDIR)/include
endif
endif

DEPS = main.cpp common.h utils/math.h pes/pes.h pes/test_surfaces.h
OBJS = main.o math.o test_surfaces.o common.o
EXTERN = 

ifeq ($(USE_MIN_FINDER), TRUE)
DEPS += swarms/swarm.h optimizers/optimizer.h
DEFINES	+= -DUSE_MIN_FINDER=$(USE_MIN_FINDER)
endif

ifeq ($(USE_TS_FINDER), TRUE)
DEPS += optimizers/ts_optimizer.h
OBJS += ts_agent.o ts_optimizer.o
DEFINES	+= -DUSE_TS_FINDER=$(USE_TS_FINDER)
endif

ifeq ($(USE_OMP), TRUE)
EXTERN += -fopenmp
CFLAGS += -fopenmp
DEFINES += -DUSE_OMP=$(USE_OMP)
endif

ifeq ($(USE_QHULL), TRUE)
DEPS += voronoi/voronoi.h 
OBJS += voronoi.o
EXTERN += -L$(QHULLDIR)/lib -lqhullcpp -lqhull_r
DEFINES	+= -DUSE_QHULL=$(USE_QHULL)
endif

ifeq ($(USE_MOLECULE), TRUE)
DEPS += molecules/molecule.h utils/xyz.h adapters/xtb_adapter.h pes/xtb_surface.h
OBJS += molecule.o xyz.o xtb_adapter.o xtb_surface.o
EXTERN += -L$(XTBDIR)/lib64 -lxtb
DEFINES	+= -DUSE_MOLECULE=$(USE_MOLECULE)
endif

execute: $(OBJS)
	${CXX} -o execute $(OBJS) $(EXTERN) $(DEFINES)

main.o: $(DEPS)
	@echo "Creating main object..."
	${CXX} ${CFLAGS} -c main.cpp $(DEFINES)

common.o: common.cpp common.h pes/pes.h
	@echo "Creating common object..."
	${CXX} ${CFLAGS} -c common.cpp $(DEFINES)

math.o: utils/math.cpp utils/math.h
	@echo "Creating math object..."
	${CXX} ${CFLAGS} -c utils/math.cpp

test_surfaces.o: pes/pes.h pes/test_surfaces.h pes/test_surfaces.cpp
	@echo "Creating test surfaces object..."
	${CXX} ${CFLAGS} -c pes/test_surfaces.cpp

ifeq ($(USE_TS_FINDER), TRUE)
ts_agent.o: pes/pes.h utils/math.h agents/ts_agent.h agents/ts_agent.cpp
	@echo "Creating TS agent object..."
	${CXX} ${CFLAGS} -c agents/ts_agent.cpp

ts_optimizer.o: pes/pes.h utils/math.h agents/ts_agent.h optimizers/ts_optimizer.h optimizers/ts_optimizer.cpp
	@echo "Creating TS optimizer object..."
	${CXX} ${CFLAGS} -c optimizers/ts_optimizer.cpp
endif

ifeq ($(USE_MOLECULE), TRUE)
molecule.o: molecules/molecule.cpp molecules/molecule.h
	@echo "Creating Molecule object..."
	${CXX} ${CFLAGS} -c molecules/molecule.cpp

xyz.o: molecules/molecule.h utils/xyz.cpp utils/xyz.h
	@echo "Creating XYZ object..."
	${CXX} ${CFLAGS} -c utils/xyz.cpp

xtb_adapter.o: adapters/xtb_adapter.h adapters/xtb_adapter.cpp utils/xyz.h molecules/molecule.h
	@echo "Creating xTB Adapter object..."
	${CXX} ${CFLAGS} -c adapters/xtb_adapter.cpp

xtb_surface.o: pes/xtb_surface.h pes/xtb_surface.cpp molecules/molecule.h adapters/xtb_adapter.h pes/pes.h
	@echo "Creating xTB PES object..."
	${CXX} ${CFLAGS} -c pes/xtb_surface.cpp
endif

ifeq ($(USE_QHULL), TRUE)
voronoi.o: voronoi/voronoi.cpp common.h
	@echo "Creating qhull object.."
	${CXX} ${CFLAGS} -c voronoi/voronoi.cpp
endif

clean:
	@echo "Cleaning up"
	rm execute main.o math.o test_surfaces.o ts_agent.o ts_optimizer.o molecule.o xyz.o common.o xtb_adapter.o xtb_surface.o voronoi.o
