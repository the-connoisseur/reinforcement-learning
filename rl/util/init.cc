#include "rl/util/init.h"

#include <atomic>
#include <vector>

#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

namespace rl::util {
namespace {
std::atomic<bool> initialized{false};
}

bool IsInitialized() { return initialized; }

void InitLoggingAndFlags(int *argc, char ***argv) {
  CHECK(!IsInitialized()) << ": Cannot initialize more than once!";
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  const std::vector<char *> positional_arguments =
      absl::ParseCommandLine(*argc, *argv);
  CHECK_LE(positional_arguments.size(), static_cast<size_t>(*argc));
  for (size_t i = 0; i < positional_arguments.size(); ++i) {
    (*argv)[i] = positional_arguments[i];
  }
  *argc = positional_arguments.size();
  initialized = true;
}

} // namespace rl::util
