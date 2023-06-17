import training
import evaluation
import time


def select_menu():
    print("Welcome to the code for the 'Projektarbeit 1' (Studiengang WWIDS-122) by Jeremy Barenkamp")
    options = ["Train custom model", "Evaluate model", "Exit script"]

    valid_choice = False

    while not valid_choice:
        print("Please choose a option" + "\n")
        for i, option in enumerate(options):
            print("(" + str((i+1)) + ") " + option)

        print()
        user_selection = input("Your selection:")

        if user_selection == "1":

            print("You have chosen wisely")
            use_standard = input("Start with standard settings? (y/n): ")

            if use_standard == "y":
                path_to_data= "./datasets/New_York_Times_labeled_dataframe_manual_cleaned.csv"
                save_path= "./datasets/Test3.pickle"
                architecture= "standard" #transformer / standard
                iterations= "6"
            else:
                path_to_data = input("Path to dataset:")

                save_path = input("Path where model should be saved:")

                architecture = input("Choose the model architecture (standard/transformer):")

                iterations = input ("Iterations (epochs) for training:")

            my_custom_model = training.custom_model()

            my_custom_model.load(path_to_data)

            start = time.time()
            my_custom_model.train(architecture, int(iterations))
            end = time.time()
            my_custom_model.save_model(save_path)

            print("\n" + "Training finished after " + str((end-start)) + " seconds")

            valid_choice = True

        elif user_selection == "2":

            myevaluatedmodel = evaluation.model_to_evaluate()

            myevaluatedmodel.load_data("./datasets/test_data_NYT.csv")

            myevaluatedmodel.load_model("./datasets/Test3.pickle", False)

            start = time.time()

            myevaluatedmodel.evaluate()

            end = time.time()

            print("\n" + "Inference finished after " + str((end - start)) + " seconds")

            valid_choice = True

        elif user_selection == "3":
            valid_choice = True
            close_program()
        else:
            valid_choice = False
            print("Wrong Input - I find your lack of faith disturbing" + "\n")


def close_program():
    print("\n" + "Bye, may the force be with you!")
    exit()


if __name__ == '__main__':
    select_menu()
