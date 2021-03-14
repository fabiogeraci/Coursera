class MusicSchool:
    students = {"Gino": [15, "653-235-345", ["Piano", "Guitar"]],
                "Talina": [28, "555-765-452", ["Cello"]],
                "Eric": [12, "583-356-223", ["Singing"]]}

    def __init__(self, working_hours, revenue):
        self.working_hours = working_hours
        self.revenue = revenue

    # Add your methods below this line
    def print_student_data(self):
        for key, value in MusicSchool.students.items():
            self.print_student(key, value)

    def print_student(self, k, v):
        print('Student: ' + k + ' who is ' + str(v[0]) + ' years old and is taking ' + str(v[2]))

    def add_student(self, new_student, age, contact, instrument):
        MusicSchool.students[new_student] = [age, contact, instrument]
        return MusicSchool.students



# Create the instance
my_school = MusicSchool(8, 15000)
my_school.print_student_data()

# Call the methods
my_school.add_student('Fabio', 45, '5555555', ['guitar'])

with open('my_school_records.txt', 'w') as data:
    for key, value in MusicSchool.students.items():
        data.write(str(key) + ': ' + str(value) + '\n')
