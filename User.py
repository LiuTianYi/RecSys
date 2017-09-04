class User:
    """
    User class

    Attributes:
        id: string
        gender: int, 1 for men and 0 for women
        age: int
        preference: array
        click_record: set, the items clicked, denoted by iid
        dislike_set: set, the disliked items, denoted by iid
    """

    def __init__(self, _id, _gender, _age, _preference):
        """Assign initial values to attributes"""
        self.id = _id
        self.gender = _gender
        self.age = _age
        self.preference = _preference
        self.click_record = set()
        self.dislike_set = set()

