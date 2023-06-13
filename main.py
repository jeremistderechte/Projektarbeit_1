import training
import evaluation
def select_menu():
    print("Welcome to the code for the 'Projektarbeit 1' (Studiengang WWIDS-122) by Jeremy Barenkamp")
    options = ["Train custom model", "Evaluate model", "Exit script"]

    valid_choice = False

    while (not valid_choice):
        print("Please choose a option" + "\n")
        for i, option in enumerate(options):
            print("(" + str((i+1)) + ") " + option)

        print()
        user_selection = input("Your selection:")

        if(user_selection == "1"):

            print("You have chosen wisely")
            use_standard = input("Start with standard settings? (y/n): ")

            if (use_standard == "y"):
                path_to_data= "./datasets/New_York_Times_labeled_dataframe_manual_cleaned.csv"
                save_path= "./datasets/New.pickle"
                architecture= "transformer"
                iterations= "6"
            else:
                path_to_data = input("Path to dataset:")

                save_path = input("Path where model should be saved:")

                architecture = input("Choose the model architecture (standard/transformer):")

                iterations = input ("Iterations (epochs) for training:")

            my_custom_model = training.custom_model()

            my_custom_model.load(path_to_data)

            my_custom_model.train(architecture, int(iterations))

            my_custom_model.save_model(save_path)

            valid_choice = True

        elif(user_selection == "2"):

            myevaluatemodel = evaluation.model_to_evaluate()

            myevaluatemodel.load_data("./datasets/test_data_NYT.csv")

            myevaluatemodel.load_model("./datasets/New.pickle", True)

            myevaluatemodel.evaluate()

            valid_choice = True

        elif(user_selection == "3"):
            valid_choice = True
            close_programm()
        else:
            valid_choice = False
            print("Wrong Input - I find your lack of faith disturbing" + "\n")

def close_programm():
    print("\n" + "Bye, may the force be with you!")
    exit()

if __name__ == '__main__':
    select_menu()
