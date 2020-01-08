# CHANGELOG

## 1.6.3 (2020-01-08)

* fix: optimize fetch samples method of sqlite reader ([c52d286](https://gitlab.com/eshard/estraces/commit/c52d286))
* maint: update README to add precisions for use with python 3.8 and build from source ([e0f2ca1](https://gitlab.com/eshard/estraces/commit/e0f2ca1))

## 1.6.2 (2019-11-26)

* fix: ETSWriter method write_trace_object working with index ([b757a70](https://gitlab.com/eshard/estraces/commit/b757a70))

## 1.6.1  (2019-11-26)

* fix: ths slicing behaves properly with metadata cache ([7bb2243](https://gitlab.com/eshard/estraces/commit/7bb2243))

## 1.6.0 (2019-11-26)

* fix: add psutil as part of estraces dependencies (which solves documentation generation failure on C ([b862873](https://gitlab.com/eshard/estraces/commit/b862873))
* fix: samples supports correct boolean indexing ([d66c7b9](https://gitlab.com/eshard/estraces/commit/d66c7b9))
* feat: add global Headers on trace header set. ([141a4ed](https://gitlab.com/eshard/estraces/commit/141a4ed))
* feat: add SQLite format reader API ([c779d53](https://gitlab.com/eshard/estraces/commit/c779d53))
* feat: improve TraceHeaderSet and Trace representations and strings ([7b409f8](https://gitlab.com/eshard/estraces/commit/7b409f8))
* feat: new in-memory metadata can be added to a Trace or TraceHeaderSet instance. ([e7bf1f3](https://gitlab.com/eshard/estraces/commit/e7bf1f3))

## 1.5.0 (2019-10-31)

* feat: change compression API to a function converter - ets writer ([6babe4d](https://gitlab.com/eshard/estraces/commit/6babe4d))

## 1.4.1 (2019-10-18)

* fix: Trace header set and trace can support a name metadata ([f36ed12](https://gitlab.com/eshard/estraces/commit/f36ed12))

## 1.4.0  (2019-10-18)

* feat: ETSWriter new APIs and improvements ([c35ade2](https://gitlab.com/eshard/estraces/commit/c35ade2))
* feat: ETSWriter support a compressed mode ([3462815](https://gitlab.com/eshard/estraces/commit/3462815))
* maint: Fix CI conda deploy target ([4ec2a2b](https://gitlab.com/eshard/estraces/commit/4ec2a2b))
