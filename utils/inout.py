from os import listdir
from os.path import join, isfile


class InOutLoop:
    """
    This class implements a loop that will iterate on every file in an 'input_folder':
    - For every file in 'input_folder' it calls a function that was registered by 'on_input' method;
    - The result of that function is passed to 'on_run' method;
    - In the end, the function that was registered by 'on_output' method is called with
      the result of 'on_run's function.
    """

    def __init__(self, input_folder='inputs', output_folder='outputs', extensions=[], debug=False):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.extensions = extensions
        self.debug = debug

    def log(self, msg):
        if self.debug:
            print(msg)

    def has_extension(self, ext, filename):
        namelen = len(filename)
        extlen = len(ext)

        return filename.find(ext, namelen - extlen, namelen) >= 0

    def check_extensions(self, filename):
        return len(self.extensions) == 0 or any(self.has_extension(ext, filename) for ext in self.extensions)

    def on_input(self, func):
        self.f_input = func

    def on_output(self, func):
        self.f_output = func

    def on_run(self, func):
        self.f_run = func

    def run(self):
        for file in listdir(self.input_folder):
            input_path = join(self.input_folder, file)
            output_path = join(self.output_folder, file)

            if isfile(input_path) and self.check_extensions(file):
                self.log('READING: {} <-'.format(file))
                result_in = self.f_input(input_path)

                self.log('PROCESSING: {} <-'.format(file))
                result_run = self.f_run(result_in)

                self.log('OUTPUT: {} <-'.format(file))
                self.f_output(output_path, result_run)
