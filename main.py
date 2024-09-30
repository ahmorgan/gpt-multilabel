import openpyxl # using openpyxl to iterate over excel file with data
from openai import OpenAI
import data_preprocessor
import csv
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix

# using the OpenAI API to prompt GPT-x models for multi-label classification of data

client = OpenAI()

# load excel file ("workbook"). data.xlsx is an excel spreadsheet with student responses
workbook = openpyxl.load_workbook(filename='data.xlsx')

# get access to excel spreadsheet
spreadsheet = workbook.active

# list of responses that will be given to the LLM
response_prompts = []

questions = [
    "How do you feel about the course so far?",
    "Explain why you selected the above choice(s).",
    "What was your biggest challenge(s) for these past modules?",
    "How did you overcome this challenge(s)? Or what steps did you start taking towards overcoming it?",
    "Do you have any current challenges in the course? If so, what are they?"
]

labels = [
    "None",
    "Python and Coding",
    "Github",
    "MySQL",
    "Assignments",
    "Quizzes",
    "Understanding requirements and instructions",
    "Learning New Material",
    "Course Structure and Materials",
    "Time Management and Motivation",
    "Group Work",
    "API",
    "Project"
]

# data preprocessing
# iterate through spreadsheet, concatenate each question followed by each student sub-response into a string that represents the full response
for row in spreadsheet.iter_rows(min_row=1, min_col=1, max_row=81, max_col=5, values_only=True):
    full_student_response = ""
    i = -1
    for value in row:
        i += 1
        # special case - some spreadsheet cells are empty, which are treated as null values
        if value is None:
            full_student_response += f"{questions[i]}: N/A"
            continue
        full_student_response += f"{questions[i]}: {value} "
    response_prompts.append(full_student_response)

# make prompts to GPT-x
llm_classifications = []
i = 0
# number of reflections to classify
num_predictions = 3
for response in response_prompts:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        # proompt engineering
        messages=[
            {"role": "system", "content": "You are a software engineering professor who has just received "
                                          "feedback responses from your students regarding their issues "
                                          "and/or experiences with your class. You seek to help them with"
                                          "their issues and ensure their success in your class."},
            {"role": "user", "content": f"Regarding the following student feedback response enclosed in quotations: "
                                        f""
                                        f"'{response}'"
                                        f""
                                        f"Choose or one more label(s) from the following list that best represents"
                                        f"the issue(s) faced by the student. Respond only with your chosen labels enclosed"
                                        f"in brackets."
                                        f""
                                        f"{labels}"}
        ]
    )
    classification = completion.choices[0].message.content
    print(classification)
    llm_classifications.append(classification)

    i += 1
    # Only generating the first ten LLM responses to save time (and a few pennies in API calls).
    if i == num_predictions:
        break

# write unprocessed classifications to a csv file (optional)
"""
with open("unprocessed_predictions.csv", "w") as output:
    i = 0
    for c in llm_classifications:
        output.write(c)
        if i != len(llm_classifications)-1:
            output.write(",\n")
        i += 1

print(llm_classifications)
"""

# encode gpt responses to create a confusion matrix out of them
gpt_preds_enc = []
for classification in llm_classifications:
    response = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    i = 0
    for label in labels:
        # gpt output is not always formatted correctly, but
        # always contains the issue classification as a substring.
        # therefore, search for substring of label in the output
        # to make classifications
        if label in classification:
            print("found")
            response[i] = 1
        i += 1
    print(response)
    gpt_preds_enc.append(response)


# data.xlsx is my personal classifications as of time of writing, will switch
# to consensus classifications when those get to me
data_preprocessor.process_data(file_name="intermediate_preds.csv")
true_preds_enc = []

# encode the test dataset in the same way
with open("intermediate_preds.csv", "r") as im:
    csv_r = csv.reader(im)
    for row in csv_r:
        response = []
        for pred in row:
            response.append(int(pred))
        true_preds_enc.append(response)
# remove now unneeded intermediate file
# optional improvement: implement with pandas dataframes instead
os.remove("intermediate_preds.csv")

true = np.array(true_preds_enc) # size num of reflections in dataset
pred = np.array(gpt_preds_enc) # size num_predictions

# only first num_predictions predictions (true is every true prediction by default)
true = true[:num_predictions]

print(f"True values:\n{true}")
print(f"Predictions:\n{pred}")

# generate confusion matrices and extract true positive, false negative,
# true negative, and false negative counts for each label
confusion_matrices = multilabel_confusion_matrix(true, pred)
print(confusion_matrices)
result = {}
x = 0
for matrix in confusion_matrices:
    # flatten confusion matrix to list
    matrix = matrix.ravel()
    print(matrix)
    # populate results with information from the label's confusion matrix
    result.update({f"{labels[x]}-tn": matrix[0].item()})
    result.update({f"{labels[x]}-fp": matrix[1].item()})
    result.update({f"{labels[x]}-fn": matrix[2].item()})
    result.update({f"{labels[x]}-tp": matrix[3].item()})
    x += 1
    if x >= len(labels):
        break

print(result)
