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
#include "Include/ImageReader.h"
#include "Include/ConfigFileParser.h"


using namespace OpenNN;
using namespace std;
using namespace cv;
/*



// Moves data through network
Vector<double> executeNetwork(Vector<double> imageData, NeuralNetwork network){

    //Make temporary storage space
    Vector<double> layerOutput;
    double perceptronOutput;

    //Grabs the MLP form the network
    MultilayerPerceptron* mlp = network.get_multilayer_perceptron_pointer();

    for(int i = 0; i < network.get_layers_number();i++) //Spool through each perceptron layer
    {
        for(int j = 0; j < mlp->get_layer_pointer(i)->get_perceptrons_number();j++)  //Calculate output for each perceptron based on previous layer's output
        {
            perceptronOutput = mlp->get_layer_pointer(i)->get_perceptron(j).calculate_output(imageData);
            layerOutput.push_back(perceptronOutput);  // Update layer input for next layer
        }
        imageData = layerOutput;
        layerOutput.clear();
    }
    return imageData;
}
*/

int main(int argc, const char** argv)
{

    try
	{
		/*//Begin Config File Parse
        ConfigFileParser configuration = ConfigFileParser("Config_File.txt");
        vector<vector<string>> layerVector = configuration.readFile();

		//istringstream inputs(layerVector[0][0]); //Grab the input configuration value (remove, needs to be determined by image)
		//int inputNumber = 0;
		//inputs >> inputNumber;

		istringstream layers(layerVector[1][0]); //Grab the layer number value
		int layerNumber = 0;
		layers >> layerNumber;

		istringstream perceptrons(layerVector[2][0]); //Grab the perceptrons per layer value
		int perceptronNumber = 0;
		perceptrons >> perceptronNumber;

		//Begin Neural Network Construction

		Vector<int> architecture; //Defines NN architecture as <# of inputs, perceptrons in layer 1, ... , perceptrons in layer n>
		architecture.push_back(151389);   //Adds # of inputs to architecture (need to remove hardcoding)
		for (int i = 0; i < layerNumber; i++)  //Add perceptrons per layer to architecture
		{
			architecture.push_back(perceptronNumber);
		}*/

        ///////--Set the Learning Rate/Error update rate--//////
        const double learning_rate = 0.3;
        const size_t iterations_num = 20;
        const size_t initial_iterations_num = 10;
        const size_t display_period = 10;

        // ----------------------------------------------------------------------//
        // -----------Get the DataSet------------//
        // ----------------------------------------------------------------------//

        //Create a new dataset object to work with
        DataSet ds;
        ds.set_data_file_name("../data/mnist_train_100.dat");
        ds.set_separator("Comma");
        ds.load_data();

        //create vector for input indices
        Vector<int> input_indices(0,784);

        for(int i = 0; i < input_indices.size(); i++){
            int x = i+1;
            input_indices.at(i) = x;
        }

        //Set targets and inputs for the data
        Variables* var_pointer = ds.get_variables_pointer();
        var_pointer->set_use(0, Variables::Target);
        var_pointer->set_use(784, Variables::Input);
        var_pointer->set_input_indices(input_indices);

        cout <<"index0: "<< var_pointer ->  write_use(0) << endl;
        cout <<"index1: "<< var_pointer ->  write_use(1) << endl;
        cout <<"index2: "<< var_pointer ->  write_use(2) << endl;
        cout <<"index783: "<< var_pointer ->  write_use(783) << endl;
        cout <<"index784: "<< var_pointer ->  write_use(783) << endl;

        //Containers for input and target info may be helpful at some point
        const Matrix<string> inputs_info = var_pointer->arrange_inputs_information();
        const Matrix<string> targets_info = var_pointer->arrange_targets_information();

        //Pointers to instances of training data may be useful
        Instances* instances_pointer = ds.get_instances_pointer();
        instances_pointer->split_random_indices();//looks at a random subset of instances

        //not sure if this is the correct scaling we want (we want all numbers between 0-1 i think)
        const Vector<Statistics<double> > input_stats = ds.scale_inputs_minimum_maximum(); //scales inputs to smaller range


        // ----------------------------------------------------------------------//
        // -----------Create NeuralNet------------//
        // ----------------------------------------------------------------------//

        //MLP for the mnist dataset to take in 28x28 image and output probabilities for 1-10
        MultilayerPerceptron mlp = MultilayerPerceptron(
                var_pointer->count_inputs_number(), 100, var_pointer->count_targets_number()
        );

		//MultilayerPerceptron mlp = MultilayerPerceptron(architecture); //Creates a multilayer perceptron with the architecture
	    NeuralNetwork neural_net1 = NeuralNetwork(mlp);	 //Creates a Neural Network with the multilayer perceptron

        //Probably good to be able to access inputs/outputs if necessary
        Inputs* inputs_pointer = neural_net1.get_inputs_pointer();
        inputs_pointer -> set_information(inputs_info); //currently empty we can fill in though

        Outputs* outputs_pointer = neural_net1.get_outputs_pointer();
        outputs_pointer -> set_information(targets_info);//currently empty we can fill in though

        //Create the scaling layer for the NN
        neural_net1.construct_scaling_layer();
        ScalingLayer* sL_pointer = neural_net1.get_scaling_layer_pointer();
        sL_pointer -> set_statistics(input_stats);
        sL_pointer -> set_scaling_method(ScalingLayer::NoScaling);//no more of this until were sure of its use

        //Construct Layer to interpret binary outputs of classification gives us a yes or no
        neural_net1.construct_probabilistic_layer();
        ProbabilisticLayer* pL_pointer = neural_net1.get_probabilistic_layer_pointer();
        pL_pointer->set_probabilistic_method(ProbabilisticLayer:: Probability);

        // ----------------------------------------------------------------------//
        // -----------Create Error Container/Loss Index------------//
        // ----------------------------------------------------------------------//

        LossIndex loss(&neural_net1, &ds);
        loss.set_error_type("NORMALIZED_SQUARED_ERROR");


        // ----------------------------------------------------------------------//
        // -----------Knowledge------------//
        // ----------------------------------------------------------------------//

        //Create the strategy and pass it the loss which it will learn, with/from?
        TrainingStrategy trainer(&loss);


        /*//set initial training method hopefully saves time
        trainer.set_initialization_type(TrainingStrategy::EVOLUTIONARY_ALGORITHM);
        EvolutionaryAlgorithm* evo_pointer = trainer.get_evolutionary_algorithm_pointer();
        evo_pointer -> set_maximum_generations_number(initial_iterations_num);*/

        //set main training method
        trainer.set_main_type(TrainingStrategy::GRADIENT_DESCENT);
        GradientDescent* grad_pointer = trainer.get_gradient_descent_pointer();
        grad_pointer -> set_maximum_iterations_number(iterations_num);
        grad_pointer -> set_display_period(display_period);

        //Do the training
        TrainingStrategy::Results training_results = trainer.perform_training();


        // ----------------------------------------------------------------------//
        // -----------Test if its working------------//
        // ----------------------------------------------------------------------//
        /*DataSet testData;
        testData.set_file_type("csv");
        testData.set_data_file_name("../data/mnist_test_10.csv");
        testData.set_separator("Comma");
        testData.load_data();*/
        instances_pointer -> set_testing();

        TestingAnalysis test_analysis(&neural_net1, &ds);


        Vector<double> classification_errors = test_analysis.calculate_classification_testing_errors();

        //OpenNN uses this confusion parameter how random our results appear
        Matrix<size_t> confusion = test_analysis.calculate_confusion();

        //Save our results to make sense of them
        ds.save("../Results/dataSet.xml");

        neural_net1.save("../Results/neural_net1.xml");
        neural_net1.save_expression("../Results/expression.txt");

        trainer.save("../Results/training_strategy.xml");
        training_results.save("../Results/training_results.dat");

        confusion.save("../Results/confusion.dat");
        classification_errors.save("../Results/classificationErrors.dat");








        /*//Show architecture
        int perceptron_count = 0;
		for(int i = 1; i < architecture.size(); i++)
		{
			perceptron_count = perceptron_count + architecture[i];	//Sums up perceptron total from architecture definition vector
		}

	    const int layer_number = neural_net1.get_layers_number();  // Number of layers for printing
		
		cout << "Perceptron Count: " << perceptron_count << endl;
		cout << "Layer Count: " << layer_number << endl;


        // Begin Image Import and Processing
        ImageReader imageReader;

        imageReader.setInputFile("MyPic.jpeg");
        vector<uchar> array = imageReader.readFile();

        // Cast uchar to double
        Vector<double> imageData;
        imageData.resize(array.size());
        for(int i=0; i < array.size(); i++)
        {
            imageData[i] = (double)array[i];
        }

        //Send data through Network
        imageData = executeNetwork(imageData, neural_net1);

        //Print NN Output
        cout << "Neural Network Result: " << endl;
        for(int i = 0; i < imageData.size();i++)
        {
            cout << imageData[i] << endl;
        }*/
        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

        return(1);
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
