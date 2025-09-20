{
  description = "AI spam classifier environment with Python, notmuch, and necessary dependencies";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    poetry2nix.url = "github:nix-community/poetry2nix";
  };

  outputs = { self, nixpkgs, poetry2nix }@inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    # python 3.11
    python = pkgs.python311;
    poetry2nixLib = poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
    aispamclassifier = poetry2nixLib.mkPoetryApplication {
      projectDir = ./.;
    };
    pythonEnv = poetry2nixLib.mkPoetryEnv {
      projectDir = ./.;
      editablePackageSources = {
        aispamclassifier = ./.;
      };
    };

    # Define the dev shell environment
    devShell = pkgs.mkShell {
      buildInputs = [
        (pkgs.poetry.override { python3 = python; })
        pythonEnv
      ];

      # Set up the Python environment
      shellHook = ''
        export VIRTUAL_ENV=".venv"
        python -m venv $VIRTUAL_ENV
        source $VIRTUAL_ENV/bin/activate
        echo "Using Python: $(python --version) from $VIRTUAL_ENV"
        echo "Welcome to the AI spam classifier environment!"
        poetry env use python
      '';
    };

  in {
    packages.default = aispamclassifier;
    devShells.${system}.default = devShell;
    apps.${system}.default = {
      type = "app";
      program = "${aispamclassifier}/bin/training";
    };
  };
}
