#ifndef RL_UTIL_INIT_H_
#define RL_UTIL_INIT_H_

namespace rl::util {

void InitLoggingAndFlags(int *argc, char ***argv);

bool IsInitialized();

} // namespace rl::util

#endif // RL_UTIL_INIT_H_
