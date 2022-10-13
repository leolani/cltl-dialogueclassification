SHELL = /bin/bash

project_dependencies ?= $(addprefix $(project_root)/, emissor \
    cltl-combot \
    cltl-requirements )

git_remote ?= https://github.com/leolani

include util/make/makefile.base.mk
include util/make/makefile.component.mk
include util/make/makefile.py.base.mk
include util/make/makefile.git.mk


build: resources/midas-da-roberta/classifier.pt


resources/midas-da-roberta/classifier.pt:
	mkdir -p resources/midas-da-roberta
	wget -O resources/midas-da-roberta/classifier.pt "https://drive.google.com/u/0/uc?id=1-33rHc9O2fM-PPaXu8I_oK5xnFwuMlN7&export=download&confirm=9iBg"


clean:
	rm resources/midas-da-roberta/classifier.pt