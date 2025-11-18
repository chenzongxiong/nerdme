import re


class Cleaner:
    def __init__(self, raw_text: str):
        self.raw_text = raw_text

    # def clean(self):
    #     # TODO: implementation of various cleaning strategies for different scenarios
    #     markdown_content = self.extract_markdown_content()
    #     if markdown_content:
    #         return markdown_content
    #     elif self.has_introductory_line():
    #         return self.remove_introductory_line().raw_text
    #     else:
    #         return self.remove_markdown_quotes_if_needed().raw_text

    # def remove_markdown_quotes_if_needed(self):
    #     """
    #     Save all content within the original Markdown block (including any internal structured code markers), while removing the externally wrapped Markdown tags.

    #     Example:
    #     ```markdown
    #     To avoid conflicts between the folder structure and our pipeline, please make sure that the <DATASET>datasets</DATASET>  have the following internal structure:
    #      * For <DATASET>CUB200-2011</DATASET>,``` cub-200-2011 └───images |    └───001.Black_footed_Albatross |           │   Black_Footed_Albatross_0001_796111 |           │   ... |    ... ```
    #      * For <DATASET>Cars196</DATASET>, please download the tar of all images, all bounding boxes, labels for both training and test  [here](http://ai.stanford.edu/~jkrause/cars/car_dataset.html) and unzip them, which is consistent with our dataloader.
    #      ``` cars196 └───car_ims |    │016180.jpg |    │   ... |    │016185.jpg └───cars_train |    │08092.jpg |    │   ... |    │08144.jpg └───cars_annos.mat  ```
    #      * For <DATASET>Stanford Online Products</DATASET>: ``` sop └───bicycle_final |   │   111085122871_0.jpg |           ... |...
    #     ```
    #     """
    #     self.raw_text = re.sub(r'```markdown\n((?:.|\n)*?)```(?=\n|$)', # r'^```markdown\n|```$',
    #                            '', self.raw_text.strip(), flags=re.MULTILINE)
    #     return self

    # def extract_markdown_content(self):
    #     match = re.search(r'```markdown\n((?:.|\n)*?)```(?=\n|$)', # r'^```markdown\n|```$',
    #                       self.raw_text, flags=re.DOTALL)
    #     return match.group(1).strip() if match else ""

    # def has_introductory_line(self):
    #     """ Example:
    #     Here is the annotated text in Markdown format:
    #     Since we evaluate on datasets in the Biomedical (`<DATASET>RELISH</DATASET>`, `<DATASET>TRECCOVID-RF</DATASET>`), Computer Science (`<DATASET>CSFCube</DATASET>`),
    #     and mixed domains (`<DATASET>SciDocs</DATASET>`) we train separate models for these domains, the sub-directories named `s2orcbiomed`, `s2orccompsci`, and
    #     `s2orcscidocs` contain config files for the models trained for each domain.
    #     Returns:

    #     """
    #     return self.raw_text.strip().startswith("Here is the annotated text in Markdown format:")

    # def remove_introductory_line(self) -> 'Cleaner':
    #     self.raw_text = re.sub(r'^Here is the annotated text in Markdown format:\s*',
    #                            '', self.raw_text.strip(), flags=re.MULTILINE)
    #     return self

    def clean(self):
        # TODO: implementation of various cleaning strategies for different scenarios
        return self.remove_markdown_quotes_if_needed().raw_text

    def remove_markdown_quotes_if_needed(self):
        # pattern = r"```.*?\n(.*?)```"
        # codes = re.findall(pattern, self.raw_text, re.DOTALL)

        # if len(codes) == 0:
        #     pass
        # # return codes[0]
        # self.raw_text = re.sub(r'^```.*?\n```$',
        #                        '', self.raw_text.strip(), flags=re.MULTILINE)
        self.raw_text = re.sub(r'^```markdown\n|```|```perl$',
                               '', self.raw_text.strip(), flags=re.MULTILINE)

        return self
