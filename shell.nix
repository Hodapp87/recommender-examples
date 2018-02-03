{ pkgs ? import <nixpkgs> {} }:
pkgs.stdenv.mkDerivation rec {
  name = "sharpestminds-web-skill-test";

  surprise = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "scikit-surprise";
      version = "1.0.5";
      doCheck = false;
      buildInputs = [ numpy six joblib011 ];
      src = pkgs.fetchurl {
        url = "mirror://pypi/s/${pname}/${name}.tar.gz";
        sha256 = "0hxgmfpvjyfi8m8xczc6kanlck3mv5j911428p7lz99qgaz71392";
      };
    };

  joblib011 = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "joblib";
      version = "0.11";
      buildInputs = [ pytest sphinx ];
      src = pkgs.fetchurl {
        url = "mirror://pypi/j/${pname}/${name}.tar.gz";
        sha256 = "11a0xl00q8kp3bmms9cn5bv03d53hp5mqf996yl335vdydnxb3vv";
      };
    };
                  
  buildInputs = [
    (pkgs.python36.withPackages
      (ps: [ps.pandas ps.numpy ps.scipy ps.scikitlearn ps.matplotlib
            surprise joblib011
            ps.pyqt4 # Needed only for matplotlib backend
            ps.jupyter # Needed for my experimentation
      ]))
  ];
}
