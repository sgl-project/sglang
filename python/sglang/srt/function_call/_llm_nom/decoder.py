from typing import Callable, Optional, Tuple

IsPendingAndText = Tuple[bool, str]
CountConsumedTokensFn = Callable[[Optional[IsPendingAndText]], int]
