"""测试包。

含 ``__init__.py`` 使其成为可导入包，配合 pyproject 中的 ``pythonpath = ["."]``，
测试里可以统一使用 ``from tests.fakes import ...``。
"""
