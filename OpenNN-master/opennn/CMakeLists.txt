



set(OPENNN_HDRS
        data_set.h
        inputs.h
    outputs.h 
    unscaling_layer.h 
    scaling_layer.h 
    probabilistic_layer.h 
    perceptron_layer.h 
    perceptron.h 
    neural_network.h 
    multilayer_perceptron.h 
    independent_parameters.h 
    conditions_layer.h 
    bounding_layer.h
        vector.h
    matrix.h
        principal_components_layer.h
        tinyxml2.h
)

set(OPENNN_SRCS
        data_set.cpp
        inputs.cpp
    outputs.cpp 
    unscaling_layer.cpp 
    scaling_layer.cpp 
    probabilistic_layer.cpp 
    perceptron_layer.cpp 
    perceptron.cpp 
    neural_network.cpp 
    multilayer_perceptron.cpp 
    independent_parameters.cpp 
    conditions_layer.cpp 
    bounding_layer.cpp
        principal_components_layer.cpp
        tinyxml2.cpp
        )

add_library(opennn ${OPENNN_SRCS})
#target_link_libraries(opennn tinyxml2)