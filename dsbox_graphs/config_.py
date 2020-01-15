import os
from d3m import utils

D3M_API_VERSION = 'v2020.1.9'
VERSION = "1.0.0"
TAG_NAME = "{git_commit}".format(git_commit=utils.current_git_commit(os.path.dirname(__file__)), )

REPOSITORY = "https://github.com/brekelma/dsbox_graphs"
PACKAGE_NAME_GRAPHS = "dsbox-graphs"

D3M_PERFORMER_TEAM = 'ISI'

if TAG_NAME:
    PACKAGE_URI_GRAPHS = "git+" + REPOSITORY + "@" + TAG_NAME
else:
    PACKAGE_URI_GRAPHS = "git+" + REPOSITORY 

PACKAGE_URI_GRAPHS = PACKAGE_URI_GRAPHS + "#egg=" + PACKAGE_NAME_GRAPHS


INSTALLATION_TYPE = 'GIT'
if INSTALLATION_TYPE == 'PYPI':
    INSTALLATION = {
        "type" : "PIP",
        "package": PACKAGE_NAME_GRAPHS,
        "version": VERSION
    }
else:
    INSTALLATION = {
        "type" : "PIP",
        "package_uri": PACKAGE_URI_GRAPHS,
    }
