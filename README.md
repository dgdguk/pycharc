# pycharc

PyCharc is a port of Matt Dale's [CHARC](https://github.com/MaterialMan/CHARC)
framework to the Python programming language, with a focus on extensibility. 

PyCharc also differs to CHARC in that it does not include a model zoo.
Instead, users should provide a subclass of `pycharc.system.System`
that allows PyCharc to interface with their model. An example of this
is provided in `esn_example.py`.

PyCharc is currently alpha software, and certain interfaces may change
as it is refined further.

PyCharc is licensed under the GNU General Public License Version 3.


