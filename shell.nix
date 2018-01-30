{ pkgs ? import <nixpkgs> {} }:
pkgs.stdenv.mkDerivation rec {
  name = "sharpestminds-web-skill-test";

  cvxpy = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "cvxpy";
      version = "0.4.11";
      doCheck = false;
      buildInputs = [ scs ecos six multiprocess scipy toolz fastcache CVXcanon ];
      src = pkgs.fetchurl {
        url = "mirror://pypi/c/${pname}/${name}.tar.gz";
        sha256 = "1lpfyv10fpvkf7famlsdlyzz5my8d5l8s0w7iqxqyyls5hgna8vm";
      };
    };  

  scs = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "scs";
      version = "2.0.2";
      buildInputs = [ numpy scipy ];
      src = pkgs.fetchurl {
        url = "mirror://pypi/s/${pname}/${name}.tar.gz";
        sha256 = "0nmnxbyf8vdn9pbn8lyzcz4nq6d63gzimyl9ksfmj9wkqiss8s64";
      };
    };  

  ecos = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "ecos";
      version = "2.0.5";
      buildInputs = [ numpy scipy ];
      src = pkgs.fetchurl {
        url = "mirror://pypi/e/${pname}/${name}.tar.gz";
        sha256 = "1srfvhvf8q1w2ccsyi0s92d19z622j1r360ym65z0hxd3kdm7gjx";
      };
    };  

  fastcache = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "fastcache";
      version = "1.0.2";
      buildInputs = [ pytest ];
      src = pkgs.fetchurl {
        url = "mirror://pypi/f/${pname}/${name}.tar.gz";
        sha256 = "1rl489zfbm2x67n7i6r7r4nhrhwk6yz3yc7x9y2rky8p95vhaw46";
      };
    };  

  CVXcanon = with pkgs.python36Packages; buildPythonPackage rec {
      name = "${pname}-${version}";
      pname = "CVXcanon";
      version = "0.1.1";
      buildInputs = [ numpy scipy ];
      doCheck = false;
      src = pkgs.fetchurl {
        url = "mirror://pypi/C/${pname}/${name}.tar.gz";
        #sha256 = "1zlvn63618gj07zmb620vcgvsvm39aa0x5bxkk2pnjqmfhbmxa7v";
        # For 0.1.1:
        sha256 = "1i8gccycawqyi59yb4kw4b28518d254i44vl5mfypqh0j8pcmmvh";
      };
    };  
              
  buildInputs = [
    (pkgs.python36.withPackages
      (ps: [ps.pandas ps.numpy ps.scipy ps.scikitlearn ps.matplotlib
            ps.pyqt4 # Needed only for matplotlib backend
            cvxpy
            CVXcanon fastcache scs ps.toolz ps.multiprocess
      ]))
  ];
}
