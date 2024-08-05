class CompareFilter:
    """
    Creates a filter object
    """
    def __init__(self, _filter=None):
        self.filter = {} if not _filter else _filter

    def equals(self):
        self.filter["compareType"] = "equal"
        return CompareFilter(self.filter)

    # def not_equals(self):
    #     self.filter["compareType"] = "equal"
    #     return CompareFilter(self.filter)

    def lesser(self):
        self.filter["compareType"] = "lesser"
        return CompareFilter(self.filter)

    def greater(self):
        self.filter["compareType"] = "greater"
        return CompareFilter(self.filter)

    def field_name(self, field_name):
        self.filter["fieldName"] = field_name
        return CompareFilter(self.filter)

    def value(self, field_value):
        self.filter["fieldValue"] = field_value
        return CompareFilter(self.filter)


if __name__ == '__main__':
    f = CompareFilter().equals().field("field_name").value("NA")
    print(f.filter)
