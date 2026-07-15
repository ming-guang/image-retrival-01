CXX = g++
SRC_DIR = src
RELEASE_DIR = ../releases
GUI_SRC = gui/image_retrieval_gui.py

# On Windows the engine is image-retrieval.exe (build_bundle.py looks for that).
ifeq ($(OS),Windows_NT)
  EXE = .exe
else
  EXE =
endif

CXXFLAGS = -std=c++11 -Wall -g -I$(SRC_DIR)
LDFLAGS = -l opencv_core -l opencv_imgcodecs -l opencv_imgproc \
          -l opencv_features2d -l opencv_flann -l opencv_highgui \
          -l opencv_objdetect

BIN = $(RELEASE_DIR)/image-retrieval$(EXE)
GUI_BIN = $(RELEASE_DIR)/image-retrieval-gui

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:.cpp=.o)

all: $(BIN) gui

# --- C++ retrieval engine ---------------------------------------------------
$(BIN): $(OBJS)
	@mkdir -p $(RELEASE_DIR)
	@echo "Linking $(BIN)..."
	$(CXX) $(OBJS) -o $(BIN) $(LDFLAGS)
	@echo "Built engine -> $(BIN)"

$(SRC_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling $<..."
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Python GUI -------------------------------------------------------------
# Installs the GUI as an executable script. Run it inside the Nix dev shell,
# where Python already has working tkinter + Pillow. This is reliable, unlike a
# PyInstaller bundle (see the gui-bundle target below).
gui: $(GUI_SRC)
	@mkdir -p $(RELEASE_DIR)
	@cp $(GUI_SRC) $(GUI_BIN)
	@chmod +x $(GUI_BIN)
	@echo "Built GUI (script) -> $(GUI_BIN)"

# Optional: pack the GUI + engine + Tcl/Tk data into ONE self-contained binary.
# build_bundle.py handles the fiddly bits (embedding the engine, locating Tk's
# data outside the Nix store) so this works where a bare `pyinstaller` did not.
# Build on the target OS -- PyInstaller cannot cross-compile.
gui-onefile: $(BIN) $(GUI_SRC)
	@command -v python3 >/dev/null 2>&1 || { echo "python3 not found (enter the dev shell)"; exit 1; }
	python3 build_bundle.py

clean:
	@echo "Cleaning up..."
	rm -f $(OBJS) $(BIN) $(GUI_BIN)
	rm -rf /tmp/pyi-build /tmp/pyi-spec

.PHONY: all clean gui gui-onefile
