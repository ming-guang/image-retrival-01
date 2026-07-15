#include "registry.h"
#include <unordered_map>

std::unordered_map<std::string, CommandFunc> registry;
std::unordered_map<std::string, DatasetCommandFunc> dataset_registry;
