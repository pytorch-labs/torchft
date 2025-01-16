# Contributing to torchft

We want to make contributing to this project as easy and transparent as possible.

## TL;DR

We appreciate all contributions. If you are interested in contributing to torchft, there are many ways to help out.
Your contributions may fall into the following categories:

- It helps the project if you can

  - Report issues that you're facing
  - Give a :+1: on issues that others reported and that are relevant to you

- Answering questions on the issue tracker, investigating bugs are very valuable contributions to the project.

- You would like to improve the documentation. This is no less important than improving the library itself! If you find
  a typo in the documentation, do not hesitate to submit a GitHub pull request.

- If you would like to fix a bug:

  - comment on the issue that you want to work on this issue
  - send a PR with your fix, see below.

- If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the
  feature with us.
- If you would like to feature a usage example in our documentation, discuss that with us in an issue.

## Issues

We use GitHub issues to track public bugs. Please follow the existing templates if possible and ensure that the
description is clear and has sufficient instructions to be able to reproduce the issue.

## Development installation

torchft is written in Python and Rust so you will need to install the Rust
toolchain. [rustup](https://rustup.rs/) is recommended as it makes using tools such as linters much
easier.

Once you have rust installed you can install the project via:

```
$ pip install -e .[dev]
```

Also see the installation instructions in [the README](./README.md).

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation and examples.
4. Ensure the test suite passes.
5. If you haven't already, complete the Contributor License Agreement ("CLA").

### Code style

`torchft` enforces a fairly strict code format with tools such as cargo fmt and black.

```shell
pip install lintrunner lintrunner-adapters
lintrunner init
lintrunner -a
```

### Tests

We use `pytest` as our testing framework. To execute a specific test, use the following command:

```sh
pytest torchft/process_group_test.py -k test_device_mesh
```

To run the Rust tests run:

```sh
cargo test
```

To run the entire suite of tests:

```sh
$ scripts/test.sh
```

### Build Docs
To build the docs run:
```sh
pip install -r docs/requirements.txt
cd docs
make livehtml
```

The docs will be built in the `docs/build/html` directory and served at http://localhost:8000.
The page will be automatically re-built as long as the process is kept running.

### Running Multiple Replica Local Job

We use torchx to run multiple worker local test jobs. You need to run the
lighthouse first and then you can use torchx to launch as many replica groups as
you want. This uses the [train_ddp.py](./train_ddp.py) script.

```sh
$ torchft_lighthouse --min_replicas 2 --join_timeout_ms 10000 &
$ torchx run -- --replicas 10
```

Once the Lighthouse has started you can view the status of all the workers at the Lighthouse dashboard.

Default address is: http://localhost:29510

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need to do this once to work on any of
Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## License

By contributing to torchft, you agree that your contributions will be licensed under the LICENSE file in the root
directory of this source tree.
