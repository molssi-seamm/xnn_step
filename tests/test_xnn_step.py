#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `xnn_step` package."""

import pytest  # noqa: F401
import xnn_step  # noqa: F401


def test_construction():
    """Just create an object and test its type."""
    result = xnn_step.Xnn()
    assert str(type(result)) == "<class 'xnn_step.xnn.Xnn'>"
