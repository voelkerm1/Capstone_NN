/****************************************************************************************************************/
/*                                                                                                              */
/*   Xavier University 2017                                                                                     */
/*                                                                                                              */
/*   CSCI 380 Opium Farm Finder                                                                                 */
/*                                                                                                              */
/*   Main File                                                                                                  */
/*                                                                                                              */
/*   Aaron Moehring, Grant Stapleton, Patrick Gemperline, Michael Voelker, Andrew Fisher, Matthew Karnes        */
/*   Under the supervision of Mikey Goldweber                                                                   */
/*                                                                                                              */
/****************************************************************************************************************/

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>

#include <stdint.h>
#include <limits.h>
#include <OpenNN-master/opennn/loss_index.h>
#include <OpenNN-master/opennn/training_strategy.h>
#include <OpenNN-master/opennn/testing_analysis.h>

// OpenNN includes

#include "vector.h"
#include "multilayer_perceptron.h"
#include "neural_network.h"

//OpenCV Includes
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

//Custom Class Includes
#include "Include/ConfigFileParser.h"


using namespace OpenNN;
using namespace std;
using namespace cv;


string getInput(string prompt){

    string response;

    cout << prompt << endl;

    cin >> response;

    return response;

}

//Checks to see if file exists
bool doesFileExist(string fileName)
{
    ifstream infile(fileName.c_str());
    return infile.good();
}

int main(int argc, const char** argv)
{

    string trainOrTest;
    cout<< "Would you like to train or test?"<< endl;
    cin >> trainOrTest;
    if (trainOrTest == "train"){
        try {
            /*
             * Build NN here based on the config files
             * We will train inside the loop bellow
             */
            NeuralNetwork * neural_net1;
            DataSet ds;

            //This gets the input file key name and initializes the file counter and the first input file

            int fileCounter = 1;

            string inputFileName;

            cout<< "Enter training filename without extension: "<< endl;
            cin>> inputFileName;

            string delimit;
            cout<< "Enter separator type as Comma, Space, Tab..., : "<< endl;
            cin>> delimit;

            string outFileName;
            cout<< "What should the output file be called?"<< endl;
            cin>> outFileName;

            string currentInFileName = inputFileName + to_string(fileCounter) + ".dat";

            cout<< currentInFileName <<endl;

            string currentOutFileName = outFileName + to_string(fileCounter) + ".xml";

            string constructionType;
            cout<< "Should a network be built from scratch or built on an existing network? Type scratch or existing. "<< endl;
            cin>> constructionType;
            ///////--Set the Learning Rate/Error update rate--//////

            const static double learning_rate = 0.3;
            const static size_t iterations_num = 1;
            const static size_t initial_iterations_num = 1;
            const static size_t display_period = 1;
            const static size_t output_num = 1;


            // ----------------------------------------------------------------------//
            // -----------Get the first DataSet------------//
            // ----------------------------------------------------------------------//

            ds.set_data_file_name(currentInFileName);
            ds.set_separator(delimit);
            cout << "loading data..." << endl;
            ds.load_data();
            cout << "data loaded success" << endl;
            cout << ds.get_data() << endl;
            //Set targets and inputs for the data
            Variables *var_pointer = ds.get_variables_pointer();
            cout << "get variables success" << endl;
            //var_pointer -> set_name(30000, "Tulips");
            cout << "set variable name success" << endl;
            //var_pointer->set_use(0, Variables::Target);
            var_pointer->set_use(30000, Variables::Target);
            //var_pointer->set_input_indices(input_indices);

            cout << "index0: " << var_pointer->write_use(0) << endl;
            cout << "index1: " << var_pointer->write_use(1) << endl;
            cout << "index2: " << var_pointer->write_use(2) << endl;
            cout << "index29999: " << var_pointer->write_use(29999) << endl;
            cout << "index30000: " << var_pointer->write_use(30000) << endl;
            cout << "index30000: " << ds.get_variable(30000) << endl;

            //Containers for input and target info may be helpful at some point
            const Matrix<string> inputs_info = var_pointer->arrange_inputs_information();
            const Matrix<string> targets_info = var_pointer->arrange_targets_information();

            //Pointers to instances of training data may be useful
            Instances *instances_pointer = ds.get_instances_pointer();
            instances_pointer->split_random_indices();//looks at a random subset of instances

            //not sure if this is the correct scaling we want (we want all numbers between 0-1 i think)
            const OpenNN::Vector<Statistics<double> > input_stats = ds.scale_inputs_minimum_maximum(); //scales inputs to smaller range


            // ----------------------------------------------------------------------//
            // -----------Create NeuralNet------------//
            // ----------------------------------------------------------------------//



            if (constructionType == "scratch") {
                cout<< "scratch selected"<<endl;
                //MLP for the mnist dataset to take in 28x28 image and output probabilities for 1-10
                MultilayerPerceptron mlp = MultilayerPerceptron(
                        var_pointer->count_inputs_number(), 100, var_pointer -> count_targets_number()
                );
                cout<< "scratch perceptron"<<endl;
                //MultilayerPerceptron mlp = MultilayerPerceptron(architecture); //Creates a multilayer perceptron with the architecture
                neural_net1 = new NeuralNetwork(mlp);    //Creates a Neural Network with the multilayer perceptron
                cout<< "scratch neural net"<<endl;
            } else {
                cout<< "scratch not selected"<<endl;

                string net_file = getInput("Enter NN .xml filename: ");
                neural_net1 = new NeuralNetwork(net_file);
                cout<< "not scratch neural net"<<endl;
            }


            //Probably good to be able to access inputs/outputs if necessary
            Inputs *inputs_pointer = neural_net1->get_inputs_pointer();
            inputs_pointer->set_information(inputs_info); //currently empty we can fill in though

            Outputs *outputs_pointer = neural_net1->get_outputs_pointer();
            outputs_pointer->set_information(targets_info);//currently empty we can fill in though

            //Create the scaling layer for the NN
            neural_net1->construct_scaling_layer();
            ScalingLayer *sL_pointer = neural_net1->get_scaling_layer_pointer();
            sL_pointer->set_statistics(input_stats);
            sL_pointer->set_scaling_method(ScalingLayer::NoScaling);//no more of this until were sure of its use

            //Construct Layer to interpret binary outputs of classification gives us a yes or no
            neural_net1->construct_probabilistic_layer();
            ProbabilisticLayer *pL_pointer = neural_net1->get_probabilistic_layer_pointer();
            pL_pointer->set_probabilistic_method(ProbabilisticLayer::Probability);




            //Checks to see if file exists and trains if it does
            while (doesFileExist(currentInFileName)) {
                cout<< "top of loop"<<endl;

                /*
                * Read in the given file and train the network
                *
                * Then when done with the given file, output to an xml that is the currentOutFileName.
                */
                // ----------------------------------------------------------------------//
                // -----------Get the DataSet------------//
                // ----------------------------------------------------------------------//

                if (fileCounter !=1){
                    ds.set_data_file_name(currentInFileName);
                    ds.set_separator(delimit);
                    cout << "loading data..." << endl;
                    ds.load_data();
                    cout << "data loaded success" << endl;
                }

                // ----------------------------------------------------------------------//
                // -----------Create Error Container/Loss Index------------//
                // ----------------------------------------------------------------------//

                LossIndex loss(neural_net1, &ds); //default is normal squared
                loss.set_error_type("SUM_SQUARED_ERROR");

                // ----------------------------------------------------------------------//
                // -----------Knowledge------------//
                // ----------------------------------------------------------------------//

                //Create the strategy and pass it the loss which it will learn, with/from?
                TrainingStrategy trainer(&loss);


                /*//set initial training method hopefully saves time
                trainer.set_initialization_type(TrainingStrategy::EVOLUTIONARY_ALGORITHM);
                EvolutionaryAlgorithm* evo_pointer = trainer.get_evolutionary_algorithm_pointer();
                evo_pointer -> set_maximum_generations_number(initial_iterations_num);*/

                //set main training method GradDesc
                trainer.set_main_type(TrainingStrategy::GRADIENT_DESCENT);
                GradientDescent *grad_pointer = trainer.get_gradient_descent_pointer();
                grad_pointer->set_maximum_iterations_number(iterations_num);
                grad_pointer->set_display_period(display_period);

                //QuasiNewton
                /*QuasiNewtonMethod* qNM = trainer.get_quasi_Newton_method_pointer();
                qNM -> set_minimum_loss_increase(1.0e-4);*/

                //Do the training
                TrainingStrategy::Results training_results = trainer.perform_training();

                //Save training strategy
                string ts_string = "../Results/training_strategy" + to_string(fileCounter) + ".xml";
                trainer.save(ts_string);

                //linReg_results.save("../Results/linReg_results3.dat");
                string training_results_string = "../Results/training_results" + to_string(fileCounter) + ".dat";
                training_results.save(training_results_string);

                //Save NN to .xml ----------------
                neural_net1->save(outFileName);
                //---------------------------------

                //Setup for next iteration
                fileCounter++;

                //get new input and output file names
                currentInFileName = inputFileName + to_string(fileCounter);

                currentOutFileName = outFileName + to_string(fileCounter);
            }
            return 0;
        }
        catch(exception& e)
        {
            cout << e.what() << endl;

            return(1);
        }
    }
    else if (trainOrTest == "test"){
        // ----------------------------------------------------------------------//
        // -----------Test if its working------------//
        // ----------------------------------------------------------------------//
        DataSet testData;
        NeuralNetwork net2;

        string testfile;
        string testDelimit;
        cout << "Enter test filename: " << endl;
        getline(cin, testfile);
        cout << "Enter separator type as Comma, Space, Tab...: " << endl;
        getline(cin, testDelimit);

        testData.set_data_file_name(testfile);
        testData.set_separator(testDelimit);
        testData.load_data();

        Variables *test_var_pointer = testData.get_variables_pointer();

        test_var_pointer->set_use(30000, Variables::Target);

        cout << "index1: " << test_var_pointer->write_use(1) << endl;
        cout << "index30000: " << test_var_pointer->write_use(30000) << endl;
        cout << "index30000: " << testData.get_variable(30000) << endl;

        Instances *test_instance_pointer = testData.get_instances_pointer();
        test_instance_pointer->set_testing();

        string testNet;
        cout << "Enter NN filename to test: " << endl;
        getline(cin, testNet);

        net2 = NeuralNetwork(testNet);

        TestingAnalysis test_analysis(&net2, &testData);

        cout << "begin testing" << endl;

        //OpenNN uses this confusion parameter how random our results appear
        Matrix<size_t> confusion = test_analysis.calculate_confusion();

        cout << "done confusion" << endl;

        //Cacluate Linear Resgression Results
        //TestingAnalysis::LinearRegressionResults linReg_results = test_analysis.perform_linear_regression_analysis();

        //Calculate Binary Classification Results
        OpenNN::Vector<double> binary_class_tests = test_analysis.calculate_binary_classification_tests();

        cout << "done binary class tests" << endl;

        OpenNN::Vector<double> classification_errors = test_analysis.calculate_classification_testing_errors();

        cout << "done class error" << endl;

        ScalingLayer* testL_pointer = net2.get_scaling_layer_pointer();
        testL_pointer->set_scaling_method(ScalingLayer::MinimumMaximum);

        string confusion_string = "../Results/confusion.dat";
        confusion.save(confusion_string);
        //classification_errors.save("../Results/classificationErrors.dat");

        string bct_string = "../Results/binary_class_tests.dat";
        binary_class_tests.save(bct_string);


    }
    else{
        cout<< "not a choice"<<endl;
    }


}


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2016 Roberto Lopez.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
