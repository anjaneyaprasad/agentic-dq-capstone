test-python:
	cd python-services && pytest

test-scala:
	cd dq-spark-project && sbt test

test: test-python test-scala
	@echo "All tests passed"