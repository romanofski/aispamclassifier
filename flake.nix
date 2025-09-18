{
  description = "AI spam classifier environment with Python, notmuch, and necessary dependencies";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system: let
    pythonEnv = nixpkgs.pkgs.python3.withPackages (pkgs: [
      pkgs.python3Packages.notmuch
      pkgs.python3Packages.scikit-learn
      pkgs.python3Packages.nltk
      pkgs.python3Packages.pandas
      pkgs.python3Packages.matplotlib
      pkgs.python3Packages.torch
      pkgs.python3Packages.transformers
      pkgs.python3Packages.datasets
    ]);

    # Define the dev shell environment
    devShell = nixpkgs.pkgs.mkShell {
      buildInputs = [
        pythonEnv
        nixpkgs.pkgs.git
        nixpkgs.pkgs.curl
        nixpkgs.pkgs.notmuch
      ];

      # Set up the Python environment
      shellHook = ''
        export PYTHONPATH=${pythonEnv}/lib/python3.9/site-packages
        export PATH=${pythonEnv}/bin:$PATH
        echo "Welcome to the AI spam classifier environment!"
      '';
    };

  in {
    # The dev shell for running the environment
    devShells.${system}.default = devShell;
  });
}
