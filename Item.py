class Item:
    """
    Item class

    Attributes:
        id: string
        topic: array
        title: array
        click_sum: int, the amount of being clicked
    """

    def __init__(self, _id, _topic, _title):
        """Assign initial values to attributes"""
        self.id = _id
        self.topic = _topic
        self.title = _title
        self.click_sum = 0
