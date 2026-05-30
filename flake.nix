{
  description = "A-Mem + HippoRAG comparison reproduction — dev shell";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        # ML wheels (torch, sentence-transformers backend) ship manylinux
        # .so files that expect to find libstdc++ / zlib on LD_LIBRARY_PATH.
        # On non-FHS Linux (e.g. NixOS), wheels fail to import without this.
        runtimeLibs = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.zlib
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            python311        # pyproject requires >= 3.11
            uv               # Python dep manager — `just sync`
            just             # task runner
            git
            gh               # PR / issue workflow
          ];

          env = {
            LD_LIBRARY_PATH = runtimeLibs;
            # Force uv to use the Nix-provided interpreter rather than
            # downloading its own — avoids glibc mismatches at runtime.
            UV_PYTHON = "${pkgs.python311}/bin/python3";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            echo "A-Mem + HippoRAG reproduction"
            echo "  python: $(python3 --version 2>&1)"
            echo "  uv:     $(uv --version 2>&1)"
            echo "  just:   $(just --version 2>&1)"
            echo ""
            echo "First time:  just sync"
            echo "Run compare: export NVIDIA_API_KEY=nvapi-… ; just compare"
            if [ -z "$NVIDIA_API_KEY" ]; then
              echo ""
              echo "Note: NVIDIA_API_KEY is not set — needed for just compare / eval / demo."
            fi
          '';
        };
      });
}
