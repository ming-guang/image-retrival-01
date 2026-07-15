{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";

  outputs = {
    self,
    nixpkgs,
  }: let
    supportedSystems = ["x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin"];
    forEachSupportedSystem = f:
      nixpkgs.lib.genAttrs supportedSystems (system:
        f {
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };
        });
  in {
    devShells = forEachSupportedSystem ({pkgs}: {
      default = pkgs.mkShell {
        venvDir = ".venv";
        packages = with pkgs; [
          gcc
          gnumake
          opencv
          ibm-plex
          tinymist
          # Python GUI: tkinter + Pillow to run it, pyinstaller to bundle it.
          # matplotlib is used by benchmark.py to plot the evaluation charts.
          (python3.withPackages (ps:
            with ps; [
              tkinter
              pillow
              pyinstaller
              matplotlib
            ]))
        ];
        shellHook = ''
          export CPLUS_INCLUDE_PATH="${pkgs.opencv}/include/opencv4:$CPLUS_INCLUDE_PATH"
          export PKG_CONFIG_PATH="${pkgs.opencv}/lib/pkgconfig:$PKG_CONFIG_PATH"
        '';
      };
    });
  };
}
