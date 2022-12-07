#include <iostream>
#include <fstream>
#include <string>

#include "network.hpp"

void
read_train_data(const std::string& path, blas::vector<blas::vector<double>>& inputs, blas::vector<blas::vector<double>>& targets);

int main()
{
    neural::network net(784, 10);

    // load training data
    blas::vector<blas::vector<double>> inputs;
    blas::vector<blas::vector<double>> targets;
    read_train_data("../dataset/", inputs, targets);

    for (int i = 0; i < 5; ++i)
    {
        for (size_t j = 0; j < inputs.size(); ++j)
        {
            std::cout << '\r' << "epoch " << i + 1 << " training: " << j << '/' << inputs.size() << std::flush;
            net.train(inputs[j], targets[j], 0.1);
        }
    }
    std::cout << std::endl;

    // test on first 60000 training data
    size_t correct = 0;
    for (size_t i = 0; i < 60000; ++i)
    {
        auto output = net.feed_forward(inputs[i]);
        auto max = std::max_element(output.begin(), output.end());
        auto target = std::max_element(targets[i].begin(), targets[i].end());
        if (*max == *target)
            ++correct;

    }

    std::cout << "accuracy: " << correct / 60000.0 * 100.0 << '%' << std::endl;

    return 0;
}

uint32_t
reverse_int(uint32_t i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
}

void
read_train_data(const std::string& path, blas::vector<blas::vector<double>>& inputs, blas::vector<blas::vector<double>>& targets)
{
    std::ifstream file(path + "/train-images-idx3-ubyte", std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "failed to open train-images.idx3-ubyte" << std::endl;
        return;
    }

    // read magic number
    uint32_t magic_number = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051)
    {
        std::cerr << "invalid magic number: " << magic_number << std::endl;
        return;
    }

    // read number of images
    uint32_t number_of_images = 0;
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = reverse_int(number_of_images);

    // read number of rows
    uint32_t number_of_rows = 0;
    file.read((char*)&number_of_rows, sizeof(number_of_rows));
    number_of_rows = reverse_int(number_of_rows);

    // read number of columns
    uint32_t number_of_columns = 0;
    file.read((char*)&number_of_columns, sizeof(number_of_columns));
    number_of_columns = reverse_int(number_of_columns);

    // read images
    inputs.resize(number_of_images);
    for (size_t i = 0; i < number_of_images; ++i)
    {   
        std::cout << '\r' << "Reading images: " << i + 1 << " / " << number_of_images << std::flush;
        inputs[i].resize(number_of_rows * number_of_columns);
        for (size_t j = 0; j < number_of_rows * number_of_columns; ++j)
        {
            unsigned char temp = 0;
            file.read((char*)&temp, sizeof(temp));
            inputs[i][j] = temp / 255.0;
        }
    }
    std::cout << std::endl;

    // read labels
    std::ifstream label_file(path + "/train-labels-idx1-ubyte", std::ios::binary);
    if (!label_file.is_open())
    {
        std::cerr << "failed to open train-labels.idx1-ubyte" << std::endl;
        return;
    }

    // read magic number
    magic_number = 0;
    label_file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049)
    {
        std::cerr << "invalid magic number: " << magic_number << std::endl;
        return;
    }

    // read number of labels
    uint32_t number_of_labels = 0;
    label_file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = reverse_int(number_of_labels);

    // read labels
    targets.resize(number_of_labels);
    for (size_t i = 0; i < number_of_labels; ++i)
    {
        std::cout << '\r' << "Reading labels: " << i + 1 << " / " << number_of_labels << std::flush;
        unsigned char temp = 0;
        label_file.read((char*)&temp, sizeof(temp));
        targets[i].resize(10);

        for (size_t j = 0; j < 10; ++j)
            targets[i][j] = 0.0;
        targets[i][temp] = 1.0;
    }
    std::cout << std::endl;
}
