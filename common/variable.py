"""
Simple data class for storing a variable, along with the necessary data to re-instantiate them
each time the page refreshes.
@author Alex Carney
"""
import json

class Variable:
    def __init__(self, letter, min_value, max_value, step_size):
        self.letter = letter
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.slider = None

    def __str__(self):
        return json.dumps({
            'letter': self.letter,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'step_size': self.step_size,
        })

