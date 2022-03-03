#include "networktables/NetworkTable.h"
#include "networktables/NetworkTableEntry.h"
#include "networktables/NetworkTableInstance.h"
#include <iostream>
#include <string>

const int TEAM = 4330;

int main(int argc, char* argv[]) {
  auto inst = nt::NetworkTableInstance::GetDefault();
  inst.StartClientTeam(TEAM);
  auto table = inst.GetTable("test");
  nt::NetworkTableEntry testEntry = table->GetEntry("test");
  testEntry.SetDefaultString("This is the default");
  std::cout << "Default entry: " << testEntry.GetString("This is the default") << std::endl;
  while (true) {
    std::cout << "Input a string: ";
    std::string testString; std::cin >> testString;
    testEntry.SetString(testString);
    std::cout << "Cool entry: " << testEntry.GetString("This is the default") << std::endl;
    std::cout << std::endl;
  }
}
