from typing import List


class CompareFilter:
    """
    Creates a filter object
    """
    def __init__(self, _filter=None):
        self._filter = {} if not _filter else _filter

    def equals(self):
        self._filter["compareType"] = "equal"
        return CompareFilter(self._filter)

    def not_equals(self):
        self._filter["compareType"] = "not_equal"
        return CompareFilter(self._filter)

    def lesser(self):
        self._filter["compareType"] = "lesser"
        return CompareFilter(self._filter)

    def greater(self):
        self._filter["compareType"] = "greater"
        return CompareFilter(self._filter)

    def gte(self):
        self._filter["compareType"] = "gte"
        return CompareFilter(self._filter)

    def lte(self):
        self._filter["compareType"] = "lte"
        return CompareFilter(self._filter)

    def field_name(self, field_name):
        self._filter["fieldName"] = field_name
        return CompareFilter(self._filter)

    def value(self, field_value):
        self._filter["fieldValue"] = field_value
        return CompareFilter(self._filter)

    @property
    def filter(self) -> dict:
        required_fields = {"fieldValue", "compareType", "fieldName"}
        for required_field in required_fields:
            if required_field not in self._filter:
                raise ValueError(f"Missing a required field. Required Fields: {required_fields}")

        return self._filter


class DomainFilters:
    def __init__(self, _filter=None):
        self._filter = {} if not _filter else _filter

    def field_name(self, field_name: str):
        self._filter["fieldName"] = field_name
        return DomainFilters(self._filter)

    def field_values(self, values: List):
        self._filter["values"] = values
        return DomainFilters(self._filter)

    @property
    def filter(self):
        required_fields = {"fieldName", "values"}
        for required_field in required_fields:
            if required_field not in self._filter:
                raise ValueError(f"Missing a required field. Required Fields: {required_fields}")

        return self._filter


if __name__ == '__main__':
    f = CompareFilter().equals().field_name("field_name").value("NA")
    print(f.filter)
