from gradio import FlaggingCallback
from gradio.components import IOComponent
from typing import Any
from llmtest import storage, constants


class MysqlLogger(FlaggingCallback):

    def __init__(self):
        pass

    def setup(self, components: list[IOComponent], flagging_dir: str = None):
        self.components = components
        self.flagging_dir = flagging_dir
        print("here in setup")

    def test(self):
        print("hello")

    def flag(
            self,
            flag_data: list[Any],
            flag_option: str = "",
            username: str = None,
    ) -> int:
        data = []
        for component, sample in zip(self.components, flag_data):
            data.append(
                component.deserialize(
                    sample,
                    None,
                    None,
                )
            )
        data.append(flag_option)
        if len(data[1]) > 0 and len(data[2]) > 0:
            storage.insert_with_rating(constants.USER_NAME, data[0], data[1], data[2], "", data[3])
        else:
            print("no data to log")

        return 1
