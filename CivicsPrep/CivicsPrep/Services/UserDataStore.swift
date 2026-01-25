import Foundation

final class UserDataStore: ObservableObject {
    @Published private(set) var bookmarks: Set<String>
    @Published private(set) var mastered: Set<String>

    init() {
        bookmarks = UserDataStore.loadSet(forKey: StorageKeys.bookmarks)
        mastered = UserDataStore.loadSet(forKey: StorageKeys.mastered)
    }

    func toggleBookmark(id: String) {
        if bookmarks.contains(id) {
            bookmarks.remove(id)
        } else {
            bookmarks.insert(id)
        }
        UserDataStore.saveSet(bookmarks, forKey: StorageKeys.bookmarks)
    }

    func toggleMastered(id: String) {
        if mastered.contains(id) {
            mastered.remove(id)
        } else {
            mastered.insert(id)
        }
        UserDataStore.saveSet(mastered, forKey: StorageKeys.mastered)
    }

    private static func loadSet(forKey key: String) -> Set<String> {
        guard let array = UserDefaults.standard.array(forKey: key) as? [String] else {
            return []
        }
        return Set(array)
    }

    private static func saveSet(_ set: Set<String>, forKey key: String) {
        UserDefaults.standard.set(Array(set), forKey: key)
    }
}
