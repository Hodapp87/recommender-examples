{ pkgs ? import <nixpkgs> {} }:
pkgs.stdenv.mkDerivation rec {
  name = "sharpestminds-web-skill-test";
  buildInputs = [
    (pkgs.python36.withPackages
      (ps: [ps.pandas ps.numpy ps.scipy ps.scikitlearn]))
  ];
}
