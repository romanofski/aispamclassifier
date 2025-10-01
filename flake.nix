{
  description =
    "AI spam classifier environment with Python, notmuch, and necessary dependencies";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, nixpkgs, pyproject-nix, uv2nix, pyproject-build-systems }@inputs:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      lib = pkgs.lib;
      # python 3.11
      python = pkgs.python312;
      pyproject-packages =
        pkgs.callPackage pyproject-nix.build.packages { inherit python; };
      pyproject-util = pkgs.callPackage pyproject-nix.build.util { };

      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      # bro I know you like overlays ...
      lockFileOverlay =
        workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      editableOverlay = workspace.mkEditablePyprojectOverlay {
        root = "$REPO_ROOT";
        members = [ "aispamclassifier" ];
      };
      buildSystemOverlay = import ./build-systems-overrides.nix;
      hacks = pkgs.callPackage pyproject-nix.build.hacks { };
      # We can't build nvidia-cuda which is a dependency of torch. First
      # off, I don't have a cuda device and secondly the nvidia-cuda
      # dependencies aren't available in nix either (judging by error
      # messages going down that route.
      pyprojectOverlay = final: prev: {
        torch = hacks.nixpkgsPrebuilt {
          from = pkgs.python312Packages.torchWithoutCuda;
          prev = prev.torch.overrideAttrs (old: {
            passthru = old.passthru // {
              dependencies =
                lib.filterAttrs (name: _: !lib.hasPrefix "nvidia" name)
                old.passthru.dependencies;
            };
          });
        };
      };

      pythonSet = pyproject-packages.overrideScope (lib.composeManyExtensions [
        pyproject-build-systems.overlays.default
        lockFileOverlay
        pyprojectOverlay
        buildSystemOverlay
        editableOverlay
      ]);

      venv = pythonSet.mkVirtualEnv "aispamclassifier" workspace.deps.default;
      # Define the dev shell environment
      devShell = pkgs.mkShell {
        packages = [ venv pkgs.uv ];
        env = {
          UV_NO_SYNC = "1";
          UV_PROJECT_ENVIRONMENT = venv;
          UV_PYTHON = pythonSet.python.interpreter;
          UV_PYTHON_DOWNLOADS = "never";
        };
        shellHook = ''
          unset PYTHONPATH
          export REPO_ROOT=$(git rev-parse --show-toplevel)
          . ${venv}/bin/activate
        '';
      };

    in {
      packages.${system}.default = venv;
      devShells.${system}.default = devShell;
      apps.${system}.default = {
        type = "app";
        program = "${venv}bin/training";
      };
    };
}
