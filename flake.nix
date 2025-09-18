{
  description = "AI spam classifier environment with Python, notmuch, and necessary dependencies";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }@inputs: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    pythonEnv = pkgs.python3.withPackages (p: with p; [
      notmuch
      scikit-learn
      nltk
      pandas
      matplotlib
      torch
      transformers
      datasets
      accelerate
      beautifulsoup4
    ]);

    # Define the dev shell environment
    devShell = pkgs.mkShell {
      buildInputs = [
        pythonEnv
        pkgs.git
        pkgs.curl
        pkgs.notmuch
      ];

      # Set up the Python environment
      shellHook = ''
        export PYTHONPATH=${pythonEnv}/lib/python3.9/site-packages
        export PATH=${pythonEnv}/bin:$PATH
        echo "Welcome to the AI spam classifier environment!"
      '';
    };

  in {
    devShells.${system}.default = devShell;
    apps.${system}.default = {
      type = "app";
      program = toString (pkgs.writeShellScript "run-classifier" ''
        exec ${pythonEnv.interpreter} ${./src/inference.py} "$@"
      '');
    };
  };
}
