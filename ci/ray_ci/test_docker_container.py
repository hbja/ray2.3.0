import sys
from typing import List
from unittest import mock

import pytest

from ci.ray_ci.docker_container import DockerContainer
from ci.ray_ci.test_base import RayCITestBase
from ci.ray_ci.utils import RAY_VERSION


class TestDockerContainer(RayCITestBase):
    cmds = []

    def test_run(self) -> None:
        def _mock_check_output(input: List[str]) -> None:
            self.cmds.append(" ".join(input))

        with mock.patch(
            "ci.ray_ci.docker_container.docker_pull", return_value=None
        ), mock.patch("subprocess.check_output", side_effect=_mock_check_output):
            container = DockerContainer("py38", "cu118", "ray")
            container.run()
            cmd = self.cmds[-1]
            assert f"ray-{RAY_VERSION}-cp38-cp38-manylinux2014_x86_64.whl" in cmd
            assert "rayproject/citemp:123-raypy38cu118base" in cmd
            assert "requirements_compiled.txt" in cmd
            assert "rayproject/ray:123456-py38-cu118" in cmd

            container = DockerContainer("py37", "cpu", "ray-ml")
            container.run()
            cmd = self.cmds[-1]
            assert f"ray-{RAY_VERSION}-cp37-cp37m-manylinux2014_x86_64.whl" in cmd
            assert "rayproject/citemp:123-ray-mlpy37cpubase" in cmd
            assert "requirements_compiled_py37.txt" in cmd
            assert "rayproject/ray-ml:123456-py37-cpu" in cmd

    def test_get_image_name(self) -> None:
        container = DockerContainer("py38", "cpu", "ray")
        assert container._get_image_names() == [
            "rayproject/ray:123456-py38-cpu",
            "rayproject/ray:123456-py38",
            "rayproject/ray:nightly-py38-cpu",
            "rayproject/ray:nightly-py38",
        ]

        container = DockerContainer("py37", "cu118", "ray-ml")
        assert container._get_image_names() == [
            "rayproject/ray-ml:123456-py37-cu118",
            "rayproject/ray-ml:123456-py37-gpu",
            "rayproject/ray-ml:123456-py37",
            "rayproject/ray-ml:nightly-py37-cu118",
            "rayproject/ray-ml:nightly-py37-gpu",
            "rayproject/ray-ml:nightly-py37",
        ]


if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
